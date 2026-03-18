#include "../../../Converter.h"
#include "../../Utils/Casts.h"
#include "../../Utils/Constants.h"
#include "../../Utils/Numerics.h"
#include "../../Utils/StmtUtils.h"
#include "./LoopUtils.h"
#include "mlir/Dialect/SCF/IR/SCF.h"

namespace cmlirc {

void CMLIRConverter::emitLoopBodyWithIV(const clang::VarDecl *inductionVar,
                                        mlir::Value ivIndex,
                                        mlir::Block *continueBlock,
                                        clang::Stmt *body) {
  mlir::OpBuilder &builder = context_manager_.Builder();
  mlir::Location loc = builder.getUnknownLoc();

  mlir::Type origType = convertType(inductionVar->getType());
  mlir::Value iv =
      origType.isIndex()
          ? ivIndex
          : mlir::arith::IndexCastOp::create(builder, loc, origType, ivIndex)
                .getResult();

  if (symbolTable.count(inductionVar))
    mlir::memref::StoreOp::create(builder, loc, iv, symbolTable[inductionVar],
                                  mlir::ValueRange{});

  loopStack.push_back({continueBlock, nullptr});
  TraverseStmt(body);
  loopStack.pop_back();
}

void CMLIRConverter::emitFullyUnrolledLoop(const SimpleLoopInfo &info,
                                           int64_t lb, int64_t ub, int64_t st,
                                           clang::Stmt *body) {
  mlir::OpBuilder &builder = context_manager_.Builder();
  mlir::Location loc = builder.getUnknownLoc();

  for (int64_t iv = lb; iv < ub; iv += st)
    emitLoopBodyWithIV(info.inductionVar, detail::indexConst(builder, loc, iv),
                       /*continueBlock=*/nullptr, body);
}

void CMLIRConverter::emitPartiallyUnrolledLoop(const SimpleLoopInfo &info,
                                               int64_t lb, int64_t ub,
                                               int64_t st, int64_t factor,
                                               clang::Stmt *body) {
  mlir::OpBuilder &builder = context_manager_.Builder();
  mlir::Location loc = builder.getUnknownLoc();

  int64_t tripCount = (ub - lb + st - 1) / st;
  int64_t numChunks = tripCount / factor;
  int64_t remainder = tripCount % factor;
  int64_t outerStride = factor * st;
  int64_t outerUBVal = lb + numChunks * outerStride;

  mlir::Value outerStep = detail::indexConst(builder, loc, outerStride);
  mlir::Value outerUB = detail::indexConst(builder, loc, outerUBVal);

  auto outerFor = mlir::scf::ForOp::create(builder, loc, info.lowerBound,
                                           outerUB, outerStep);
  {
    mlir::OpBuilder::InsertionGuard guard(builder);

    outerFor.getBody()->back().erase();
    builder.setInsertionPointToEnd(outerFor.getBody());

    for (int64_t j = 0; j < factor; ++j) {
      mlir::Value ivIndex = outerFor.getInductionVar();
      if (j != 0) {
        mlir::Value off = detail::indexConst(builder, loc, j * st);
        ivIndex =
            mlir::arith::AddIOp::create(builder, loc, ivIndex, off).getResult();
      }
      emitLoopBodyWithIV(info.inductionVar, ivIndex, outerFor.getBody(), body);
    }
    detail::ensureYield(builder, loc, outerFor.getBody());
  }
  builder.setInsertionPointAfter(outerFor);

  for (int64_t j = 0; j < remainder; ++j) {
    int64_t ivConst = lb + (numChunks * factor + j) * st;
    emitLoopBodyWithIV(info.inductionVar,
                       detail::indexConst(builder, loc, ivConst),
                       /*continueBlock=*/nullptr, body);
  }
}

void CMLIRConverter::emitPlainForLoop(const SimpleLoopInfo &info,
                                      clang::Stmt *body) {
  mlir::OpBuilder &builder = context_manager_.Builder();
  mlir::Location loc = builder.getUnknownLoc();

  auto forOp = mlir::scf::ForOp::create(builder, loc, info.lowerBound,
                                        info.upperBound, info.step);

  forOp.getBody()->back().erase();
  builder.setInsertionPointToEnd(forOp.getBody());

  mlir::Value iv = forOp.getInductionVar();

  if (!info.isIncrementing) {
    mlir::Value ub1 =
        mlir::arith::SubIOp::create(builder, loc, info.upperBound,
                                    detail::indexConst(builder, loc, 1))
            .getResult();
    mlir::Value off =
        mlir::arith::SubIOp::create(builder, loc, iv, info.lowerBound)
            .getResult();
    iv = mlir::arith::SubIOp::create(builder, loc, ub1, off).getResult();
  }

  emitLoopBodyWithIV(info.inductionVar, iv, forOp.getBody(), body);

  detail::ensureYield(builder, loc, forOp.getBody());
  builder.setInsertionPointAfter(forOp);
}

void CMLIRConverter::emitWhileStyleForLoop(clang::ForStmt *forStmt) {
  mlir::OpBuilder &builder = context_manager_.Builder();
  mlir::Location loc = builder.getUnknownLoc();

  if (forStmt->getInit())
    TraverseStmt(forStmt->getInit());

  const bool hasBreak = detail::stmtHasBreakInLoop(forStmt);
  const bool hasContinue = detail::stmtHasContinueInLoop(forStmt);
  const bool hasReturn = detail::stmtHasReturnInLoop(forStmt);
  const bool needsGuard = hasBreak || hasContinue || hasReturn;

  mlir::Type funcRetType;
  {
    auto res = currentFunc.getFunctionType().getResults();
    if (!res.empty())
      funcRetType = res[0];
  }

  auto allocBool = [&]() -> mlir::Value {
    auto a = mlir::memref::AllocaOp::create(
        builder, loc, mlir::MemRefType::get({}, builder.getI1Type()));
    mlir::memref::StoreOp::create(builder, loc,
                                  detail::boolConst(builder, loc, false),
                                  a.getResult(), mlir::ValueRange{});
    return a.getResult();
  };

  mlir::Value breakFlag = hasBreak ? allocBool() : mlir::Value{};
  mlir::Value continueFlag = hasContinue ? allocBool() : mlir::Value{};

  if (hasReturn && funcRetType) {
    mlir::Value initRetVal =
        mlir::arith::ConstantOp::create(builder, loc, funcRetType,
                                        builder.getZeroAttr(funcRetType))
            .getResult();
    mlir::Value initKeepGoing = detail::boolConst(builder, loc, true);

    auto whileOp = mlir::scf::WhileOp::create(
        builder, loc,
        /*resultTypes=*/mlir::TypeRange{funcRetType},
        /*operands=*/mlir::ValueRange{initRetVal, initKeepGoing},
        [&](mlir::OpBuilder &b, mlir::Location l, mlir::ValueRange args) {
          mlir::OpBuilder::InsertionGuard g(builder);
          builder.setInsertionPointToEnd(b.getInsertionBlock());

          mlir::Value cond =
              forStmt->getCond()
                  ? detail::toBool(builder, loc,
                                   generateExpr(forStmt->getCond()))
                  : detail::boolConst(builder, loc, true);

          mlir::Value finalCond = detail::andi(builder, loc, cond, args[1]);

          if (breakFlag) {
            mlir::Value notBroke = detail::noti(
                builder, loc,
                mlir::memref::LoadOp::create(builder, loc, breakFlag)
                    .getResult());
            finalCond = detail::andi(builder, loc, finalCond, notBroke);
          }

          mlir::scf::ConditionOp::create(builder, loc, finalCond,
                                         mlir::ValueRange{args[0]});
        },
        [&](mlir::OpBuilder &b, mlir::Location l, mlir::ValueRange) {
          mlir::scf::YieldOp::create(
              b, l, mlir::ValueRange{initRetVal, initKeepGoing});
        });

    mlir::Block *afterBlock = &whileOp.getAfter().front();
    afterBlock->back().erase();
    builder.setInsertionPointToEnd(afterBlock);

    mlir::Value prevRetVal = afterBlock->getArgument(0);

    mlir::Value iterRetFlag =
        mlir::memref::AllocaOp::create(
            builder, loc, mlir::MemRefType::get({}, builder.getI1Type()))
            .getResult();
    mlir::memref::StoreOp::create(builder, loc,
                                  detail::boolConst(builder, loc, false),
                                  iterRetFlag, mlir::ValueRange{});

    mlir::Value iterRetSlot =
        mlir::memref::AllocaOp::create(builder, loc,
                                       mlir::MemRefType::get({}, funcRetType))
            .getResult();

    loopStack.push_back(LoopContext{&whileOp.getBefore().front(), afterBlock,
                                    breakFlag, continueFlag, iterRetFlag,
                                    iterRetSlot});

    // Body
    auto *compound =
        llvm::dyn_cast_or_null<clang::CompoundStmt>(forStmt->getBody());
    if (needsGuard && compound) {
      for (clang::Stmt *s : compound->body()) {
        mlir::Value guard = detail::buildGuard(builder, loc, breakFlag,
                                               continueFlag, iterRetFlag);
        detail::emitGuarded(builder, loc, guard, [&] { TraverseStmt(s); });
      }
    } else {
      TraverseStmt(forStmt->getBody());
    }

    loopStack.pop_back();

    if (forStmt->getInc()) {
      mlir::Value notRet = detail::noti(
          builder, loc,
          mlir::memref::LoadOp::create(builder, loc, iterRetFlag).getResult());
      mlir::Value incGuard = notRet;
      if (breakFlag) {
        mlir::Value notBroke = detail::noti(
            builder, loc,
            mlir::memref::LoadOp::create(builder, loc, breakFlag).getResult());
        incGuard = detail::andi(builder, loc, incGuard, notBroke);
      }
      detail::emitGuarded(builder, loc, incGuard,
                          [&] { TraverseStmt(forStmt->getInc()); });
    }

    if (continueFlag)
      mlir::memref::StoreOp::create(builder, loc,
                                    detail::boolConst(builder, loc, false),
                                    continueFlag, mlir::ValueRange{});

    mlir::Value didReturn =
        mlir::memref::LoadOp::create(builder, loc, iterRetFlag).getResult();
    mlir::Value loadedRetVal =
        mlir::memref::LoadOp::create(builder, loc, iterRetSlot).getResult();
    mlir::Value newRetVal =
        mlir::arith::SelectOp::create(builder, loc, didReturn, loadedRetVal,
                                      prevRetVal)
            .getResult();
    mlir::Value newKeepGoing = detail::noti(builder, loc, didReturn);

    mlir::scf::YieldOp::create(builder, loc,
                               mlir::ValueRange{newRetVal, newKeepGoing});

    builder.setInsertionPointAfter(whileOp);
    mlir::func::ReturnOp::create(builder, loc,
                                 mlir::ValueRange{whileOp.getResult(0)});
    return;
  }

  auto whileOp = mlir::scf::WhileOp::create(
      builder, loc, mlir::TypeRange{}, mlir::ValueRange{},
      [&](mlir::OpBuilder &b, mlir::Location l, mlir::ValueRange) {
        mlir::OpBuilder::InsertionGuard g(builder);
        builder.setInsertionPointToEnd(b.getInsertionBlock());
        mlir::Value cond =
            forStmt->getCond()
                ? detail::toBool(builder, loc, generateExpr(forStmt->getCond()))
                : detail::boolConst(builder, loc, true);
        if (breakFlag) {
          mlir::Value notBroke =
              detail::noti(builder, loc,
                           mlir::memref::LoadOp::create(builder, loc, breakFlag)
                               .getResult());
          cond = detail::andi(builder, loc, cond, notBroke);
        }
        mlir::scf::ConditionOp::create(builder, loc, cond, mlir::ValueRange{});
      },
      [&](mlir::OpBuilder &b, mlir::Location l, mlir::ValueRange) {
        mlir::scf::YieldOp::create(b, l, mlir::ValueRange{});
      });

  mlir::Block *afterBlock = &whileOp.getAfter().front();
  afterBlock->back().erase();
  builder.setInsertionPointToEnd(afterBlock);

  loopStack.push_back(LoopContext{&whileOp.getBefore().front(),
                                  afterBlock,
                                  breakFlag,
                                  continueFlag,
                                  {},
                                  {}});

  auto *compound =
      llvm::dyn_cast_or_null<clang::CompoundStmt>(forStmt->getBody());
  if (needsGuard && compound) {
    for (clang::Stmt *s : compound->body()) {
      mlir::Value guard =
          detail::buildGuard(builder, loc, breakFlag, continueFlag, {});
      detail::emitGuarded(builder, loc, guard, [&] { TraverseStmt(s); });
    }
  } else {
    TraverseStmt(forStmt->getBody());
  }

  loopStack.pop_back();

  if (forStmt->getInc()) {
    mlir::Block *cur = builder.getInsertionBlock();
    builder.setInsertionPointToEnd(cur);
    if (cur->empty() || !cur->back().hasTrait<mlir::OpTrait::IsTerminator>()) {
      mlir::Value runInc = hasBreak ? detail::noti(builder, loc,
                                                   mlir::memref::LoadOp::create(
                                                       builder, loc, breakFlag)
                                                       .getResult())
                                    : detail::boolConst(builder, loc, true);
      detail::emitGuarded(builder, loc, runInc,
                          [&] { TraverseStmt(forStmt->getInc()); });
    }
  }

  if (continueFlag)
    mlir::memref::StoreOp::create(builder, loc,
                                  detail::boolConst(builder, loc, false),
                                  continueFlag, mlir::ValueRange{});

  detail::ensureYield(builder, loc, builder.getInsertionBlock());
  builder.setInsertionPointAfter(whileOp);
}

bool CMLIRConverter::TraverseForStmt(clang::ForStmt *forStmt) {
  if (!currentFunc)
    return true;

  if (detail::stmtHasBreakInLoop(forStmt) ||
      detail::stmtHasContinueInLoop(forStmt) ||
      detail::stmtHasReturnInLoop(forStmt)) {
    emitWhileStyleForLoop(forStmt);
    return true;
  }

  mlir::OpBuilder &builder = context_manager_.Builder();
  mlir::Location loc = builder.getUnknownLoc();

  auto info =
      detail::analyseForLoop(forStmt, builder, loc, [this](clang::Expr *e) {
        return generateExpr(e);
      });

  if (!info) {
    emitWhileStyleForLoop(forStmt);
    return true;
  }

  if (forStmt->getInit())
    TraverseStmt(forStmt->getInit());

  clang::SourceManager &SM = context_manager_.ClangContext().getSourceManager();
  uint32_t forLine = SM.getSpellingLineNumber(forStmt->getForLoc());
  auto hintIt = loop_hints_.find(forLine);

  if (hintIt != loop_hints_.end()) {
    const LoopHints &h = hintIt->second;
    auto lb = detail::getInt(info->lowerBound);
    auto ub = detail::getInt(info->upperBound);
    auto st = detail::getInt(info->step);

    if (lb && ub && st && *st > 0) {
      int64_t tripCount = (*ub - *lb + *st - 1) / *st;

      if (h.unrollFull ||
          (h.unrollCount && (int64_t)*h.unrollCount >= tripCount)) {
        emitFullyUnrolledLoop(*info, *lb, *ub, *st, forStmt->getBody());
        return true;
      }

      if (h.unrollCount && (int64_t)*h.unrollCount > 1) {
        emitPartiallyUnrolledLoop(*info, *lb, *ub, *st, (int64_t)*h.unrollCount,
                                  forStmt->getBody());
        return true;
      }
    }
  }

  emitPlainForLoop(*info, forStmt->getBody());
  return true;
}

} // namespace cmlirc
