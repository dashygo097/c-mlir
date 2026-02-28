#include "../../../Converter.h"
#include "../../Utils/Casts.h"
#include "./CFUtils.h"
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

  loopStack_.push_back({continueBlock, nullptr});
  TraverseStmt(body);
  loopStack_.pop_back();
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
    mlir::OpBuilder::InsertionGuard g(builder);
    builder.setInsertionPointToStart(outerFor.getBody());

    for (int64_t j = 0; j < factor; ++j) {
      // iv = outerIV + j*st
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
  builder.setInsertionPointToStart(forOp.getBody());

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

  auto whileOp = mlir::scf::WhileOp::create(
      builder, loc, mlir::TypeRange{}, mlir::ValueRange{},
      [&](mlir::OpBuilder &b, mlir::Location l, mlir::ValueRange args) {
        mlir::Value cond = forStmt->getCond()
                               ? convertToBool(generateExpr(forStmt->getCond()))
                               : detail::boolConst(b, l, true);
        mlir::scf::ConditionOp::create(b, l, cond, mlir::ValueRange{});
      },
      [&](mlir::OpBuilder &b, mlir::Location l, mlir::ValueRange args) {
        mlir::scf::YieldOp::create(b, l, mlir::ValueRange{});
      });

  mlir::Block *afterBlock = &whileOp.getAfter().front();

  afterBlock->back().erase();

  builder.setInsertionPointToEnd(afterBlock);
  loopStack_.push_back({&whileOp.getBefore().front(), afterBlock});
  TraverseStmt(forStmt->getBody());
  if (forStmt->getInc())
    generateExpr(forStmt->getInc());
  loopStack_.pop_back();

  detail::ensureYield(builder, loc, afterBlock);
  builder.setInsertionPointAfter(whileOp);
}

bool CMLIRConverter::TraverseForStmt(clang::ForStmt *forStmt) {
  if (!currentFunc)
    return true;

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
    auto lb = detail::getConstantInt(info->lowerBound);
    auto ub = detail::getConstantInt(info->upperBound);
    auto st = detail::getConstantInt(info->step);

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
