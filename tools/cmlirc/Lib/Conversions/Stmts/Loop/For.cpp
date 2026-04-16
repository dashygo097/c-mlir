#include "../../../Converter.h"
#include "../../Utils/Casts.h"
#include "../../Utils/Constants.h"
#include "../../Utils/Numerics.h"
#include "../../Utils/StmtUtils.h"
#include "./LoopUtils.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"

namespace cmlirc {

void CMLIRConverter::emitLoopBodyWithIV(const clang::VarDecl *inductionVar,
                                        mlir::Value ivIndex,
                                        mlir::Block *continueBlock,
                                        clang::Stmt *body) {
  mlir::OpBuilder &builder = contextManager.Builder();
  mlir::Location loc = builder.getUnknownLoc();

  mlir::Type origType = convertType(inductionVar->getType());
  mlir::Value iv =
      origType.isIndex()
          ? ivIndex
          : mlir::arith::IndexCastOp::create(builder, loc, origType, ivIndex)
                .getResult();

  if (symbolTable.count(inductionVar)) {
    mlir::memref::StoreOp::create(builder, loc, iv, symbolTable[inductionVar],
                                  mlir::ValueRange{});
  }

  loopStack.push_back({continueBlock, nullptr});
  TraverseStmt(body);
  loopStack.pop_back();
}

void CMLIRConverter::emitPlainForLoop(clang::ForStmt *forStmt,
                                      const SimpleLoopInfo &info,
                                      clang::Stmt *body) {
  mlir::OpBuilder &builder = contextManager.Builder();
  mlir::Location loc = builder.getUnknownLoc();

  auto forOp = mlir::scf::ForOp::create(builder, loc, info.lowerBound,
                                        info.upperBound, info.step);

  clang::SourceManager &sm = contextManager.ClangContext().getSourceManager();
  uint32_t forLine = sm.getSpellingLineNumber(forStmt->getForLoc());

  if (loopHintMap.count(forLine)) {
    const LoopHints &h = loopHintMap[forLine];
    if (h.unrollDisable) {
      forOp->setAttr("nounroll", builder.getUnitAttr());
    } else if (h.unrollFull) {
      forOp->setAttr("unroll", builder.getUnitAttr());
    } else if (h.unrollCount) {
      forOp->setAttr("unroll_count", builder.getI32IntegerAttr(*h.unrollCount));
    }
    if (h.vectorize) {
      forOp->setAttr("vectorize", builder.getUnitAttr());
      uint32_t width = h.vectorizeWidth ? *h.vectorizeWidth : 4;
      forOp->setAttr("vectorize_width", builder.getI32IntegerAttr(width));
    }
  }

  forOp.getBody()->back().erase();
  builder.setInsertionPointToEnd(forOp.getBody());

  mlir::Value iv = forOp.getInductionVar();

  if (!info.isIncrementing) {
    mlir::Value ub1 =
        mlir::arith::SubIOp::create(builder, loc, info.upperBound,
                                    utils::indexConst(builder, loc, 1))
            .getResult();
    mlir::Value off =
        mlir::arith::SubIOp::create(builder, loc, iv, info.lowerBound)
            .getResult();
    iv = mlir::arith::SubIOp::create(builder, loc, ub1, off).getResult();
  }

  emitLoopBodyWithIV(info.inductionVar, iv, forOp.getBody(), body);

  utils::ensureYield(builder, loc, forOp.getBody());
  builder.setInsertionPointAfter(forOp);
}

void CMLIRConverter::emitWhileStyleForLoop(clang::ForStmt *forStmt) {
  mlir::OpBuilder &builder = contextManager.Builder();
  mlir::Location loc = builder.getUnknownLoc();

  if (forStmt->getInit()) {
    TraverseStmt(forStmt->getInit());
  }

  const bool hasBreak = utils::stmtHasBreakInLoop(forStmt);
  const bool hasContinue = utils::stmtHasContinueInLoop(forStmt);
  const bool hasReturn = utils::stmtHasReturnInLoop(forStmt);
  const bool needsGuard = hasBreak || hasContinue || hasReturn;

  mlir::Type funcRetType;
  {
    auto res = currentFunc.getFunctionType().getResults();
    if (!res.empty()) {
      funcRetType = res[0];
    }
  }

  auto allocBool = [&]() -> mlir::Value {
    auto a = mlir::memref::AllocaOp::create(
        builder, loc, mlir::MemRefType::get({}, builder.getI1Type()));
    mlir::memref::StoreOp::create(builder, loc,
                                  utils::boolConst(builder, loc, false),
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
    mlir::Value initKeepGoing = utils::boolConst(builder, loc, true);

    auto whileOp = mlir::scf::WhileOp::create(
        builder, loc, mlir::TypeRange{funcRetType},
        mlir::ValueRange{initRetVal, initKeepGoing},
        [&](mlir::OpBuilder &b, mlir::Location l, mlir::ValueRange args) {
          mlir::OpBuilder::InsertionGuard g(builder);
          builder.setInsertionPointToEnd(b.getInsertionBlock());

          mlir::Value cond =
              forStmt->getCond()
                  ? utils::toBool(builder, loc,
                                  generateExpr(forStmt->getCond()))
                  : utils::boolConst(builder, loc, true);

          mlir::Value finalCond = utils::andi(builder, loc, cond, args[1]);

          if (breakFlag) {
            mlir::Value notBroke = utils::noti(
                builder, loc,
                mlir::memref::LoadOp::create(builder, loc, breakFlag)
                    .getResult());
            finalCond = utils::andi(builder, loc, finalCond, notBroke);
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
                                  utils::boolConst(builder, loc, false),
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
        mlir::Value guard = utils::buildGuard(builder, loc, breakFlag,
                                              continueFlag, iterRetFlag);
        utils::emitGuarded(builder, loc, guard, [&] { TraverseStmt(s); });
      }
    } else {
      TraverseStmt(forStmt->getBody());
    }

    loopStack.pop_back();

    if (forStmt->getInc()) {
      mlir::Value notRet = utils::noti(
          builder, loc,
          mlir::memref::LoadOp::create(builder, loc, iterRetFlag).getResult());
      mlir::Value incGuard = notRet;
      if (breakFlag) {
        mlir::Value notBroke = utils::noti(
            builder, loc,
            mlir::memref::LoadOp::create(builder, loc, breakFlag).getResult());
        incGuard = utils::andi(builder, loc, incGuard, notBroke);
      }
      utils::emitGuarded(builder, loc, incGuard,
                         [&] { TraverseStmt(forStmt->getInc()); });
    }

    if (continueFlag) {
      mlir::memref::StoreOp::create(builder, loc,
                                    utils::boolConst(builder, loc, false),
                                    continueFlag, mlir::ValueRange{});
    }

    mlir::Value didReturn =
        mlir::memref::LoadOp::create(builder, loc, iterRetFlag).getResult();
    mlir::Value loadedRetVal =
        mlir::memref::LoadOp::create(builder, loc, iterRetSlot).getResult();
    mlir::Value newRetVal =
        mlir::arith::SelectOp::create(builder, loc, didReturn, loadedRetVal,
                                      prevRetVal)
            .getResult();
    mlir::Value newKeepGoing = utils::noti(builder, loc, didReturn);

    mlir::scf::YieldOp::create(builder, loc,
                               mlir::ValueRange{newRetVal, newKeepGoing});

    builder.setInsertionPointAfter(whileOp);
    mlir::func::ReturnOp::create(builder, loc,
                                 mlir::ValueRange{whileOp.getResult(0)});
    return;
  } else if (hasReturn && !funcRetType) {
    mlir::Value initKeepGoing = utils::boolConst(builder, loc, true);

    auto whileOp = mlir::scf::WhileOp::create(
        builder, loc, mlir::TypeRange{builder.getI1Type()},
        mlir::ValueRange{initKeepGoing},
        [&](mlir::OpBuilder &b, mlir::Location l, mlir::ValueRange args) {
          mlir::OpBuilder::InsertionGuard g(builder);
          builder.setInsertionPointToEnd(b.getInsertionBlock());

          mlir::Value cond =
              forStmt->getCond()
                  ? utils::toBool(builder, loc,
                                  generateExpr(forStmt->getCond()))
                  : utils::boolConst(builder, loc, true);

          mlir::Value finalCond = utils::andi(builder, loc, cond, args[0]);

          if (breakFlag) {
            mlir::Value notBroke = utils::noti(
                builder, loc,
                mlir::memref::LoadOp::create(builder, loc, breakFlag)
                    .getResult());
            finalCond = utils::andi(builder, loc, finalCond, notBroke);
          }

          mlir::scf::ConditionOp::create(builder, loc, finalCond,
                                         mlir::ValueRange{args[0]});
        },
        [&](mlir::OpBuilder &b, mlir::Location l, mlir::ValueRange args) {
          mlir::scf::YieldOp::create(b, l, args);
        });

    mlir::Block *afterBlock = &whileOp.getAfter().front();
    afterBlock->back().erase();
    builder.setInsertionPointToEnd(afterBlock);

    mlir::Value iterRetFlag =
        mlir::memref::AllocaOp::create(
            builder, loc, mlir::MemRefType::get({}, builder.getI1Type()))
            .getResult();
    mlir::memref::StoreOp::create(
        builder, loc, utils::boolConst(builder, loc, false), iterRetFlag);

    loopStack.push_back(LoopContext{&whileOp.getBefore().front(),
                                    afterBlock,
                                    breakFlag,
                                    continueFlag,
                                    iterRetFlag,
                                    {}});

    auto *compound =
        llvm::dyn_cast_or_null<clang::CompoundStmt>(forStmt->getBody());
    if (needsGuard && compound) {
      for (clang::Stmt *s : compound->body()) {
        mlir::Value guard = utils::buildGuard(builder, loc, breakFlag,
                                              continueFlag, iterRetFlag);
        utils::emitGuarded(builder, loc, guard, [&] { TraverseStmt(s); });
      }
    } else {
      TraverseStmt(forStmt->getBody());
    }

    loopStack.pop_back();

    if (forStmt->getInc()) {
      mlir::Value notRet = utils::noti(
          builder, loc,
          mlir::memref::LoadOp::create(builder, loc, iterRetFlag).getResult());
      mlir::Value incGuard = notRet;
      if (breakFlag) {
        mlir::Value notBroke = utils::noti(
            builder, loc,
            mlir::memref::LoadOp::create(builder, loc, breakFlag).getResult());
        incGuard = utils::andi(builder, loc, incGuard, notBroke);
      }
      utils::emitGuarded(builder, loc, incGuard,
                         [&] { TraverseStmt(forStmt->getInc()); });
    }

    if (continueFlag) {
      mlir::memref::StoreOp::create(
          builder, loc, utils::boolConst(builder, loc, false), continueFlag);
    }

    mlir::Value didReturn =
        mlir::memref::LoadOp::create(builder, loc, iterRetFlag).getResult();
    mlir::Value newKeepGoing = utils::noti(builder, loc, didReturn);

    mlir::scf::YieldOp::create(builder, loc, mlir::ValueRange{newKeepGoing});

    builder.setInsertionPointAfter(whileOp);

    return;
  }

  auto whileOp = mlir::scf::WhileOp::create(
      builder, loc, mlir::TypeRange{}, mlir::ValueRange{},
      [&](mlir::OpBuilder &b, mlir::Location l, mlir::ValueRange) {
        mlir::OpBuilder::InsertionGuard g(builder);
        builder.setInsertionPointToEnd(b.getInsertionBlock());
        mlir::Value cond =
            forStmt->getCond()
                ? utils::toBool(builder, loc, generateExpr(forStmt->getCond()))
                : utils::boolConst(builder, loc, true);
        if (breakFlag) {
          mlir::Value notBroke =
              utils::noti(builder, loc,
                          mlir::memref::LoadOp::create(builder, loc, breakFlag)
                              .getResult());
          cond = utils::andi(builder, loc, cond, notBroke);
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
          utils::buildGuard(builder, loc, breakFlag, continueFlag, {});
      utils::emitGuarded(builder, loc, guard, [&] { TraverseStmt(s); });
    }
  } else {
    TraverseStmt(forStmt->getBody());
  }

  loopStack.pop_back();

  if (forStmt->getInc()) {
    mlir::Block *cur = builder.getInsertionBlock();
    builder.setInsertionPointToEnd(cur);
    if (cur->empty() || !cur->back().hasTrait<mlir::OpTrait::IsTerminator>()) {
      mlir::Value runInc = hasBreak ? utils::noti(builder, loc,
                                                  mlir::memref::LoadOp::create(
                                                      builder, loc, breakFlag)
                                                      .getResult())
                                    : utils::boolConst(builder, loc, true);
      utils::emitGuarded(builder, loc, runInc,
                         [&] { TraverseStmt(forStmt->getInc()); });
    }
  }

  if (continueFlag) {
    mlir::memref::StoreOp::create(builder, loc,
                                  utils::boolConst(builder, loc, false),
                                  continueFlag, mlir::ValueRange{});
  }

  utils::ensureYield(builder, loc, builder.getInsertionBlock());
  builder.setInsertionPointAfter(whileOp);
}

auto CMLIRConverter::TraverseForStmt(clang::ForStmt *forStmt) -> bool {
  if (!currentFunc) {
    return true;
  }

  if (utils::stmtHasBreakInLoop(forStmt) ||
      utils::stmtHasContinueInLoop(forStmt) ||
      utils::stmtHasReturnInLoop(forStmt)) {
    emitWhileStyleForLoop(forStmt);
    return true;
  }

  mlir::OpBuilder &builder = contextManager.Builder();
  mlir::Location loc = builder.getUnknownLoc();

  auto info =
      utils::analyseForLoop(forStmt, builder, loc,
                            [this](clang::Expr *e) { return generateExpr(e); });

  if (!info) {
    emitWhileStyleForLoop(forStmt);
    return true;
  }

  if (forStmt->getInit()) {
    TraverseStmt(forStmt->getInit());
  }

  emitPlainForLoop(forStmt, *info, forStmt->getBody());

  return true;
}

} // namespace cmlirc
