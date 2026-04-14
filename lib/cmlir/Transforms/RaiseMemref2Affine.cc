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
  if (auto addOp = val.getDefiningOp<mlir::arith::AddIOp>()) {
    return isAffineIndex(addOp.getLhs()) && isAffineIndex(addOp.getRhs());
  }
  if (auto subOp = val.getDefiningOp<mlir::arith::SubIOp>()) {
    return isAffineIndex(subOp.getLhs()) && isAffineIndex(subOp.getRhs());
  }
  if (auto mulOp = val.getDefiningOp<mlir::arith::MulIOp>()) {
    return isAffineIndex(mulOp.getLhs()) && isAffineIndex(mulOp.getRhs());
  }
  if (mlir::isa<mlir::BlockArgument>(val)) {
    return true;
  }
  return false;
}

static auto buildAffineExpr(mlir::Value val,
                            llvm::SmallVectorImpl<mlir::Value> &operands,
                            mlir::PatternRewriter &rewriter)
    -> mlir::AffineExpr {
  if (auto constOp = val.getDefiningOp<mlir::arith::ConstantOp>()) {
    if (auto intAttr = mlir::dyn_cast<mlir::IntegerAttr>(constOp.getValue())) {
      return rewriter.getAffineConstantExpr(intAttr.getInt());
    }
  }
  if (auto addOp = val.getDefiningOp<mlir::arith::AddIOp>()) {
    return buildAffineExpr(addOp.getLhs(), operands, rewriter) +
           buildAffineExpr(addOp.getRhs(), operands, rewriter);
  }
  if (auto subOp = val.getDefiningOp<mlir::arith::SubIOp>()) {
    return buildAffineExpr(subOp.getLhs(), operands, rewriter) -
           buildAffineExpr(subOp.getRhs(), operands, rewriter);
  }
  if (auto mulOp = val.getDefiningOp<mlir::arith::MulIOp>()) {
    return buildAffineExpr(mulOp.getLhs(), operands, rewriter) *
           buildAffineExpr(mulOp.getRhs(), operands, rewriter);
  }

  for (size_t i = 0; i < operands.size(); ++i) {
    if (operands[i] == val) {
      return rewriter.getAffineDimExpr(i);
    }
  }
  operands.push_back(val);
  return rewriter.getAffineDimExpr(operands.size() - 1);
}

static auto buildAffineMapAndOperands(
    mlir::PatternRewriter &rewriter, mlir::OperandRange indices,
    llvm::SmallVectorImpl<mlir::Value> &operands) -> mlir::AffineMap {
  llvm::SmallVector<mlir::AffineExpr> exprs;
  for (mlir::Value index : indices) {
    exprs.push_back(buildAffineExpr(index, operands, rewriter));
  }
  return mlir::AffineMap::get(operands.size(), 0, exprs, rewriter.getContext());
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

    mlir::affine::fullyComposeAffineMapAndOperands(&map, &mapOperands);

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

    mlir::affine::fullyComposeAffineMapAndOperands(&map, &mapOperands);

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
