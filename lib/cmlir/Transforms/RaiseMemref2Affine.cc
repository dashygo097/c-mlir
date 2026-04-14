#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#define GEN_PASS_DEF_RAISEMEMREF2AFFINEPASS
#include "cmlir/Transforms/Passes.h.inc"

namespace cmlir {

static auto isAffineIndex(mlir::Value val) -> bool {
  if (val.getDefiningOp<mlir::arith::ConstantOp>()) {
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

struct RaiseArithAdd2Affine
    : public mlir::OpRewritePattern<mlir::arith::AddIOp> {
  using OpRewritePattern::OpRewritePattern;
  auto matchAndRewrite(mlir::arith::AddIOp op,
                       mlir::PatternRewriter &rewriter) const
      -> mlir::LogicalResult override {
    if (!op.getType().isIndex()) {
      return mlir::failure();
    }
    auto map = mlir::AffineMap::get(
        2, 0, rewriter.getAffineDimExpr(0) + rewriter.getAffineDimExpr(1));
    rewriter.replaceOpWithNewOp<mlir::affine::AffineApplyOp>(
        op, map, mlir::ValueRange{op.getLhs(), op.getRhs()});
    return mlir::success();
  }
};

struct RaiseArithSub2Affine
    : public mlir::OpRewritePattern<mlir::arith::SubIOp> {
  using OpRewritePattern::OpRewritePattern;
  auto matchAndRewrite(mlir::arith::SubIOp op,
                       mlir::PatternRewriter &rewriter) const
      -> mlir::LogicalResult override {
    if (!op.getType().isIndex()) {
      return mlir::failure();
    }
    auto map = mlir::AffineMap::get(
        2, 0, rewriter.getAffineDimExpr(0) - rewriter.getAffineDimExpr(1));
    rewriter.replaceOpWithNewOp<mlir::affine::AffineApplyOp>(
        op, map, mlir::ValueRange{op.getLhs(), op.getRhs()});
    return mlir::success();
  }
};

struct RaiseArithMul2Affine
    : public mlir::OpRewritePattern<mlir::arith::MulIOp> {
  using OpRewritePattern::OpRewritePattern;
  auto matchAndRewrite(mlir::arith::MulIOp op,
                       mlir::PatternRewriter &rewriter) const
      -> mlir::LogicalResult override {
    if (!op.getType().isIndex()) {
      return mlir::failure();
    }
    auto map = mlir::AffineMap::get(
        2, 0, rewriter.getAffineDimExpr(0) * rewriter.getAffineDimExpr(1));
    rewriter.replaceOpWithNewOp<mlir::affine::AffineApplyOp>(
        op, map, mlir::ValueRange{op.getLhs(), op.getRhs()});
    return mlir::success();
  }
};

struct RaiseMemrefLoad2AffineLoadPattern
    : public mlir::OpRewritePattern<mlir::memref::LoadOp> {
  using OpRewritePattern::OpRewritePattern;
  auto matchAndRewrite(mlir::memref::LoadOp loadOp,
                       mlir::PatternRewriter &rewriter) const
      -> mlir::LogicalResult override {
    for (mlir::Value index : loadOp.getIndices()) {
      if (!isAffineIndex(index)) {
        return mlir::failure();
      }
    }
    rewriter.replaceOpWithNewOp<mlir::affine::AffineLoadOp>(
        loadOp, loadOp.getMemRef(), loadOp.getIndices());
    return mlir::success();
  }
};

struct RaiseMemrefStore2AffineStorePattern
    : public mlir::OpRewritePattern<mlir::memref::StoreOp> {
  using OpRewritePattern::OpRewritePattern;
  auto matchAndRewrite(mlir::memref::StoreOp storeOp,
                       mlir::PatternRewriter &rewriter) const
      -> mlir::LogicalResult override {
    for (mlir::Value index : storeOp.getIndices()) {
      if (!isAffineIndex(index)) {
        return mlir::failure();
      }
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

    patterns.add<RaiseArithAdd2Affine>(op->getContext());
    patterns.add<RaiseArithSub2Affine>(op->getContext());
    patterns.add<RaiseArithMul2Affine>(op->getContext());
    patterns.add<RaiseMemrefLoad2AffineLoadPattern>(op->getContext());
    patterns.add<RaiseMemrefStore2AffineStorePattern>(op->getContext());

    if (mlir::failed(mlir::applyPatternsGreedily(op, std::move(patterns)))) {
      signalPassFailure();
      return;
    }
  }
};

auto createRaiseMemref2AffinePass() -> std::unique_ptr<mlir::Pass> {
  return std::make_unique<RaiseMemref2AffinePass>();
}

} // namespace cmlir
