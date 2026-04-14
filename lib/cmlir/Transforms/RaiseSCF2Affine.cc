#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#define GEN_PASS_DEF_RAISESCF2AFFINEPASS
#include "cmlir/Transforms/Passes.h.inc"

namespace cmlir {

static auto resolveIndexCastChain(mlir::Value v) -> mlir::Value {
  auto outerCast = v.getDefiningOp<mlir::arith::IndexCastOp>();
  if (!outerCast) {
    return v;
  }
  mlir::Value mid = outerCast.getIn();
  if (mid.getType().isIndex()) {
    return resolveIndexCastChain(mid);
  }
  auto innerCast = mid.getDefiningOp<mlir::arith::IndexCastOp>();
  if (!innerCast) {
    return v;
  }
  mlir::Value original = innerCast.getIn();
  if (!original.getType().isIndex()) {
    return v;
  }
  return resolveIndexCastChain(original);
}

static auto getBoundInfo(mlir::Value bound, mlir::MLIRContext *ctx,
                         mlir::AffineMap &map,
                         llvm::SmallVectorImpl<mlir::Value> &operands) -> bool {
  if (auto constOp = bound.getDefiningOp<mlir::arith::ConstantOp>()) {
    if (auto intAttr = mlir::dyn_cast<mlir::IntegerAttr>(constOp.getValue())) {
      map = mlir::AffineMap::getConstantMap(intAttr.getInt(), ctx);
      return true;
    }
  }

  mlir::Value resolved = resolveIndexCastChain(bound);

  if (mlir::affine::isValidSymbol(resolved)) {
    map = mlir::AffineMap::get(0, 1, mlir::getAffineSymbolExpr(0, ctx));
    operands.push_back(resolved);
    return true;
  }

  if (mlir::affine::isValidDim(resolved)) {
    map = mlir::AffineMap::get(1, 0, mlir::getAffineDimExpr(0, ctx));
    operands.push_back(resolved);
    return true;
  }

  return false;
}

struct RaiseSCFFor2AffineForPattern
    : public mlir::OpRewritePattern<mlir::scf::ForOp> {
  using mlir::OpRewritePattern<mlir::scf::ForOp>::OpRewritePattern;

  auto matchAndRewrite(mlir::scf::ForOp forOp,
                       mlir::PatternRewriter &rewriter) const
      -> mlir::LogicalResult override {

    mlir::MLIRContext *ctx = rewriter.getContext();
    mlir::Location loc = forOp.getLoc();

    mlir::AffineMap lbMap, ubMap;
    llvm::SmallVector<mlir::Value> lbOperands, ubOperands;

    if (!getBoundInfo(forOp.getLowerBound(), ctx, lbMap, lbOperands)) {
      return mlir::failure();
    }
    if (!getBoundInfo(forOp.getUpperBound(), ctx, ubMap, ubOperands)) {
      return mlir::failure();
    }

    int64_t stepVal = 0;
    if (auto stepConst =
            forOp.getStep().getDefiningOp<mlir::arith::ConstantOp>()) {
      if (auto intAttr =
              mlir::dyn_cast<mlir::IntegerAttr>(stepConst.getValue())) {
        stepVal = intAttr.getInt();
      } else {
        return mlir::failure();
      }
    } else {
      return mlir::failure();
    }

    auto affineFor = mlir::affine::AffineForOp::create(
        rewriter, loc, lbOperands, lbMap, ubOperands, ubMap, stepVal,
        forOp.getInitArgs(),
        [](mlir::OpBuilder &, mlir::Location, mlir::Value,
           mlir::ValueRange) -> void {});

    mlir::Block *affineBody = affineFor.getBody();
    mlir::Block *scfBody = forOp.getBody();

    if (!affineBody->empty() &&
        affineBody->back().hasTrait<mlir::OpTrait::IsTerminator>()) {
      rewriter.eraseOp(&affineBody->back());
    }

    mlir::IRMapping mapping;
    mapping.map(forOp.getInductionVar(), affineFor.getInductionVar());
    for (auto [scfArg, affineArg] :
         llvm::zip(forOp.getRegionIterArgs(), affineFor.getRegionIterArgs())) {
      mapping.map(scfArg, affineArg);
    }

    rewriter.setInsertionPointToEnd(affineBody);

    for (auto &op : scfBody->without_terminator()) {
      rewriter.clone(op, mapping);
    }

    auto scfYield = mlir::cast<mlir::scf::YieldOp>(scfBody->getTerminator());
    llvm::SmallVector<mlir::Value> yieldOps;
    for (mlir::Value v : scfYield.getOperands()) {
      yieldOps.push_back(mapping.lookupOrDefault(v));
    }
    mlir::affine::AffineYieldOp::create(rewriter, scfYield.getLoc(), yieldOps);

    rewriter.replaceOp(forOp, affineFor.getResults());
    return mlir::success();
  }
};

struct RaiseSCF2AffinePass
    : public impl::RaiseSCF2AffinePassBase<RaiseSCF2AffinePass> {

  void runOnOperation() override {
    auto op = getOperation();
    mlir::RewritePatternSet patterns(op->getContext());

    patterns.add<RaiseSCFFor2AffineForPattern>(op->getContext());

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

auto createRaiseSCF2AffinePass() -> std::unique_ptr<mlir::Pass> {
  return std::make_unique<RaiseSCF2AffinePass>();
}

} // namespace cmlir
