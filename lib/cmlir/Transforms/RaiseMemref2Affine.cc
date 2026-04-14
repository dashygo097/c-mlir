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

static auto getConstIntValue(mlir::Value val) -> std::optional<int64_t> {
  if (auto constOp = val.getDefiningOp<mlir::arith::ConstantOp>()) {
    if (auto intAttr = mlir::dyn_cast<mlir::IntegerAttr>(constOp.getValue())) {
      return intAttr.getInt();
    }
  }
  return std::nullopt;
}

static auto buildAffineMapAndOperands(
    mlir::PatternRewriter &rewriter, mlir::OperandRange indices,
    llvm::SmallVectorImpl<mlir::Value> &operands) -> mlir::AffineMap {

  llvm::SmallVector<mlir::AffineExpr> exprs;
  unsigned dimCount = 0;

  for (mlir::Value index : indices) {
    if (auto constVal = getConstIntValue(index)) {
      exprs.push_back(rewriter.getAffineConstantExpr(*constVal));
    } else {
      operands.push_back(index);
      exprs.push_back(rewriter.getAffineDimExpr(dimCount++));
    }
  }

  return mlir::AffineMap::get(dimCount, 0, exprs, rewriter.getContext());
}

struct RaiseMemrefLoad2AffineLoadPattern
    : public mlir::OpRewritePattern<mlir::memref::LoadOp> {
  using mlir::OpRewritePattern<mlir::memref::LoadOp>::OpRewritePattern;

  auto matchAndRewrite(mlir::memref::LoadOp loadOp,
                       mlir::PatternRewriter &rewriter) const
      -> mlir::LogicalResult override {

    for (mlir::Value index : loadOp.getIndices()) {
      if (!isAffineIndex(index)) {
        return mlir::failure();
      }
    }

    llvm::SmallVector<mlir::Value> mapOperands;
    mlir::AffineMap map =
        buildAffineMapAndOperands(rewriter, loadOp.getIndices(), mapOperands);

    rewriter.replaceOpWithNewOp<mlir::affine::AffineLoadOp>(
        loadOp, loadOp.getMemRef(), map, mapOperands);

    return mlir::success();
  }
};

struct RaiseMemrefStore2AffineStorePattern
    : public mlir::OpRewritePattern<mlir::memref::StoreOp> {
  using mlir::OpRewritePattern<mlir::memref::StoreOp>::OpRewritePattern;

  auto matchAndRewrite(mlir::memref::StoreOp storeOp,
                       mlir::PatternRewriter &rewriter) const
      -> mlir::LogicalResult override {

    for (mlir::Value index : storeOp.getIndices()) {
      if (!isAffineIndex(index)) {
        return mlir::failure();
      }
    }

    llvm::SmallVector<mlir::Value> mapOperands;
    mlir::AffineMap map =
        buildAffineMapAndOperands(rewriter, storeOp.getIndices(), mapOperands);

    rewriter.replaceOpWithNewOp<mlir::affine::AffineStoreOp>(
        storeOp, storeOp.getValue(), storeOp.getMemRef(), map, mapOperands);

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
