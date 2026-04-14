#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Dominance.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#define GEN_PASS_DEF_RAISEMEMREF2AFFINEPASS
#include "cmlir/Transforms/Passes.h.inc"

namespace cmlir {

static auto isAffineIndex(mlir::Value val) -> bool {
  if (val.getDefiningOp<mlir::arith::ConstantOp>()) {
    return true;
  }
  if (val.getDefiningOp<mlir::arith::ConstantIndexOp>()) {
    return true;
  }

  if (val.getDefiningOp<mlir::affine::AffineApplyOp>()) {
    return true;
  }

  if (auto blockArg = mlir::dyn_cast<mlir::BlockArgument>(val)) {
    if (mlir::isa<mlir::affine::AffineForOp>(
            blockArg.getOwner()->getParentOp())) {
      return true;
    }
    if (mlir::isa<mlir::func::FuncOp>(blockArg.getOwner()->getParentOp())) {
      return true;
    }
  }

  return false;
}

static auto allIndicesAffineValid(const mlir::OperandRange &indices) -> bool {
  for (auto index : indices) {
    if (!isAffineIndex(index)) {
      return false;
    }
  }
  return true;
}

struct RaiseMemrefLoad2AffineLoadPattern
    : public mlir::OpRewritePattern<mlir::memref::LoadOp> {
  using mlir::OpRewritePattern<mlir::memref::LoadOp>::OpRewritePattern;

  auto matchAndRewrite(mlir::memref::LoadOp loadOp,
                       mlir::PatternRewriter &rewriter) const
      -> mlir::LogicalResult override {

    if (!allIndicesAffineValid(loadOp.getIndices())) {
      return mlir::failure();
    }

    rewriter.replaceOpWithNewOp<mlir::affine::AffineLoadOp>(
        loadOp, loadOp.getMemRef(), loadOp.getIndices());

    return mlir::success();
  }
};

struct RaiseMemrefStore2AffineStorePattern
    : public mlir::OpRewritePattern<mlir::memref::StoreOp> {
  using mlir::OpRewritePattern<mlir::memref::StoreOp>::OpRewritePattern;

  auto matchAndRewrite(mlir::memref::StoreOp storeOp,
                       mlir::PatternRewriter &rewriter) const
      -> mlir::LogicalResult override {

    if (!allIndicesAffineValid(storeOp.getIndices())) {
      return mlir::failure();
    }

    rewriter.replaceOpWithNewOp<mlir::affine::AffineStoreOp>(
        storeOp, storeOp.getValue(), storeOp.getMemRef(), storeOp.getIndices());

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
