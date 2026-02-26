#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/IR/Dominance.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#define GEN_PASS_DEF_FMAPASS
#include "cmlir/Transforms/Passes.h.inc"

namespace cmlir {

// %0 = memref.alloca() : memref<4x4xf32>
// %1 = memref.load %0[%i, %j] : memref<4x4xf32>
// =>
// %0 = memref.alloca() : memref<4x4xf32>
// %1 = affine.load %0[%i, %j] : memref<4x4xf32>
struct FuseMultiplyAddPattern
    : public mlir::OpRewritePattern<mlir::arith::AddFOp> {
  using mlir::OpRewritePattern<mlir::arith::AddFOp>::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::arith::AddFOp addOp,
                  mlir::PatternRewriter &rewriter) const override {
    auto mulOp = addOp.getLhs().getDefiningOp<mlir::arith::MulFOp>();
    if (!mulOp)
      return mlir::failure();
    rewriter.replaceOpWithNewOp<mlir::math::FmaOp>(
        addOp, mulOp.getLhs(), mulOp.getRhs(), addOp.getRhs());
    return mlir::success();
  }
};

struct FMAPass : public impl::FMAPassBase<FMAPass> {

  void runOnOperation() override {
    auto op = getOperation();
    mlir::RewritePatternSet patterns(op->getContext());

    patterns.add<FuseMultiplyAddPattern>(&getContext());

    if (mlir::failed(mlir::applyPatternsGreedily(op, std::move(patterns)))) {
      signalPassFailure();
      return;
    }

    op->walk([](mlir::Operation *op) {
      if (mlir::isOpTriviallyDead(op)) {
        op->erase();
      }
    });
  }
};

std::unique_ptr<mlir::Pass> createFMAPass() {
  return std::make_unique<FMAPass>();
}

} // namespace cmlir
