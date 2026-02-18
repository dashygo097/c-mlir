#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/IR/Dominance.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#define GEN_PASS_DEF_FMAPASS
#include "cmlir/Transforms/Passes.h.inc"

namespace cmlir {

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

struct FMAPass
    : public mlir::PassWrapper<FMAPass,
                               mlir::OperationPass<mlir::func::FuncOp>> {

  void runOnOperation() override {
    mlir::func::FuncOp func = getOperation();
    mlir::RewritePatternSet patterns(&getContext());

    patterns.add<FuseMultiplyAddPattern>(&getContext());

    if (mlir::failed(mlir::applyPatternsGreedily(func, std::move(patterns)))) {
      signalPassFailure();
      return;
    }

    func.walk([](mlir::Operation *op) {
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
