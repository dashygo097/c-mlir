#include "../ControlFlow/LoopAnalysis.h"
#include "../Lowering.h"
#include "clang-mlir/Casts.h"
#include "clang-mlir/Constants.h"
#include "clang-mlir/Numerics.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"

namespace cmlir::frontend {

namespace {

struct SimpleLoopInfo {
  const clang::VarDecl *inductionVar;
  mlir::Type inductionType;
  mlir::Value lowerBound;
  mlir::Value upperBound;
  mlir::Value step;
  bool isIncrementing;
};

class ForLoopEmitter {
public:
  ForLoopEmitter(ASTLowering &lowering, clang_mlir::Context &context,
                 FunctionState &function, const LoopHintMap &loopHints)
      : lowering(lowering), context(context), function(function),
        loopHints(loopHints) {}

  auto emit(clang::ForStmt *stmt, const SimpleLoopInfo &info) -> void;

private:
  auto emitBody(const SimpleLoopInfo &info, mlir::Value ivIndex,
                clang::Stmt *body) -> void;

  ASTLowering &lowering;
  clang_mlir::Context &context;
  FunctionState &function;
  const LoopHintMap &loopHints;
};

auto ForLoopEmitter::emitBody(const SimpleLoopInfo &info, mlir::Value ivIndex,
                              clang::Stmt *body) -> void {
  mlir::OpBuilder &builder = context.builder();
  mlir::Location loc = builder.getUnknownLoc();

  mlir::Type origType = info.inductionType;
  mlir::Value iv =
      origType.isIndex()
          ? ivIndex
          : mlir::arith::IndexCastOp::create(builder, loc, origType, ivIndex)
                .getResult();

  if (function.locals.count(info.inductionVar)) {
    mlir::memref::StoreOp::create(builder, loc, iv,
                                  function.locals[info.inductionVar],
                                  mlir::ValueRange{});
  }

  lowering.emitStatement(body);
}

auto ForLoopEmitter::emit(clang::ForStmt *forStmt, const SimpleLoopInfo &info)
    -> void {
  mlir::OpBuilder &builder = context.builder();
  mlir::Location loc = builder.getUnknownLoc();

  auto forOp = mlir::scf::ForOp::create(builder, loc, info.lowerBound,
                                        info.upperBound, info.step);

  clang::SourceManager &sm = context.ast().getSourceManager();
  uint32_t forLine = sm.getSpellingLineNumber(forStmt->getForLoc());

  if (loopHints.count(forLine)) {
    const LoopHints &h = loopHints.find(forLine)->second;
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
    mlir::Value ub1 = clang_mlir::sub(builder, loc, info.upperBound,
                                      clang_mlir::indexConst(builder, loc, 1));
    mlir::Value off = clang_mlir::sub(builder, loc, iv, info.lowerBound);
    iv = clang_mlir::sub(builder, loc, ub1, off);
  }

  emitBody(info, iv, forStmt->getBody());

  mlir::scf::YieldOp::create(builder, loc);
  builder.setInsertionPointAfter(forOp);
}

} // namespace

auto ASTLowering::emitFor(clang::ForStmt *stmt) -> bool {
  if (control::hasControlExit(stmt->getBody())) {
    if (stmt->getInit() && !emitStatement(stmt->getInit())) {
      return false;
    }
    return emitCFGLoop(stmt->getCond(), stmt->getBody(), stmt->getInc(), false);
  }
  auto &builder = context.builder();
  auto loc = context.source().location(stmt->getForLoc());
  auto plan = control::analyseForLoop(stmt, context.ast());
  if (!plan) {
    if (stmt->getInit() && !emitStatement(stmt->getInit())) {
      return false;
    }
    return emitCFGLoop(stmt->getCond(), stmt->getBody(), stmt->getInc(), false);
  }
  if (!emitStatement(stmt->getInit())) {
    return false;
  }
  auto inductionType = types.convertType(plan->induction->getType());
  auto address =
      clang_mlir::Address::get(function.locals[plan->induction], inductionType);
  if (mlir::failed(address)) {
    return false;
  }
  auto init = clang_mlir::toIndex(builder, loc, address->load(builder, loc));
  auto bound = clang_mlir::toIndex(builder, loc, generateExpr(plan->bound));
  if (!init || !bound) {
    return false;
  }
  SimpleLoopInfo info{plan->induction,
                      inductionType,
                      plan->increasing ? init : bound,
                      plan->increasing ? bound : init,
                      clang_mlir::indexConst(builder, loc, plan->step),
                      plan->increasing};
  if (plan->comparison == clang::BO_LE) {
    info.upperBound = clang_mlir::addi(builder, loc, info.upperBound, 1);
  }
  if (plan->comparison == clang::BO_GT) {
    info.lowerBound = clang_mlir::addi(builder, loc, info.lowerBound, 1);
  }
  if (!plan->increasing) {
    info.upperBound = clang_mlir::addi(builder, loc, info.upperBound, 1);
  }
  ForLoopEmitter(*this, context, function, loopHintMap).emit(stmt, info);
  return !context.source().failed();
}

} // namespace cmlir::frontend
