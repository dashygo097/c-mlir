#include "mlir/IR/Dominance.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#define GEN_PASS_DEF_RAISESCF2AFFINEPASS
#include "cmlir/Transforms/Passes.h.inc"

namespace cmlir {

struct RaiseSCF2AffinePass
    : public impl::RaiseSCF2AffinePassBase<RaiseSCF2AffinePass> {

  void runOnOperation() override {
    auto op = getOperation();
    mlir::RewritePatternSet patterns(op->getContext());

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

std::unique_ptr<mlir::Pass> createRaiseSCF2AffinePass() {
  return std::make_unique<RaiseSCF2AffinePass>();
}

} // namespace cmlir
