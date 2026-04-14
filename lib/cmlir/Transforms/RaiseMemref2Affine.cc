#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Dominance.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#define GEN_PASS_DEF_RAISEMEMREF2AFFINEPASS
#include "cmlir/Transforms/Passes.h.inc"

namespace cmlir {

struct RaiseMemrefLoad2AffineLoadPattern
    : public mlir::OpRewritePattern<mlir::memref::LoadOp> {
  using mlir::OpRewritePattern<mlir::memref::LoadOp>::OpRewritePattern;

  auto matchAndRewrite(mlir::memref::LoadOp loadOp,
                       mlir::PatternRewriter &rewriter) const
      -> mlir::LogicalResult override {
    return mlir::success();
  }
};

struct RaiseMemrefStore2AffineStorePattern
    : public mlir::OpRewritePattern<mlir::memref::StoreOp> {
  using mlir::OpRewritePattern<mlir::memref::StoreOp>::OpRewritePattern;

  auto matchAndRewrite(mlir::memref::StoreOp storeOp,
                       mlir::PatternRewriter &rewriter) const
      -> mlir::LogicalResult override {
    return mlir::success();
  }
};

struct RaiseMemref2AffinePass
    : public impl::RaiseMemref2AffinePassBase<RaiseMemref2AffinePass> {

  void runOnOperation() override {
    auto op = getOperation();
    mlir::RewritePatternSet patterns(op->getContext());

    patterns.add<RaiseMemrefLoad2AffineLoadPattern>(op->getContext());
    patterns.add<RaiseMemrefStore2AffineStorePattern>(op->getContext());

    if (mlir::failed(mlir::applyPatternsGreedily(op, std::move(patterns)))) {
      signalPassFailure();
      return;
    }

    op->walk([](mlir::Operation *op) -> void {
      if (mlir::isOpTriviallyDead(op)) {
        op->erase();
      }
    });
  }
};

auto createRaiseMemref2AffinePass() -> std::unique_ptr<mlir::Pass> {
  return std::make_unique<RaiseMemref2AffinePass>();
}

} // namespace cmlir
