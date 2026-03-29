#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Dominance.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
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

static auto isLegalAffineIndex(mlir::Value v) -> bool {
  if (mlir::affine::isValidDim(v) || mlir::affine::isValidSymbol(v)) {
    return true;
}

  if (auto blockArg = llvm::dyn_cast<mlir::BlockArgument>(v)) {
    mlir::Operation *parentOp = blockArg.getOwner()->getParentOp();
    if (auto forOp = llvm::dyn_cast_or_null<mlir::scf::ForOp>(parentOp)) {
      if (blockArg == forOp.getInductionVar()) {
        return true;
}
}
  }

  return false;
}

static auto getBoundInfo(mlir::Value bound, mlir::MLIRContext *ctx,
                         mlir::AffineMap &map,
                         llvm::SmallVectorImpl<mlir::Value> &operands) -> bool {
  if (auto c = bound.getDefiningOp<mlir::arith::ConstantIndexOp>()) {
    map = mlir::AffineMap::getConstantMap(c.value(), ctx);
    return true;
  }

  mlir::Value resolved = resolveIndexCastChain(bound);

  if (mlir::affine::isValidSymbol(resolved)) {
    // affine_map<()[s0] -> (s0)>
    map = mlir::AffineMap::get(/*dimCount=*/0, /*symbolCount=*/1,
                               mlir::getAffineSymbolExpr(0, ctx));
    operands.push_back(resolved);
    return true;
  }

  if (mlir::affine::isValidDim(resolved)) {
    // affine_map<(d0) -> (d0)>
    map = mlir::AffineMap::get(/*dimCount=*/1, /*symbolCount=*/0,
                               mlir::getAffineDimExpr(0, ctx));
    operands.push_back(resolved);
    return true;
  }

  return false;
}

struct RaiseSCFFor2AffineForPattern
    : public mlir::OpRewritePattern<mlir::scf::ForOp> {
  using mlir::OpRewritePattern<mlir::scf::ForOp>::OpRewritePattern;

  auto
  matchAndRewrite(mlir::scf::ForOp forOp,
                  mlir::PatternRewriter &rewriter) const -> mlir::LogicalResult override {

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

    auto stepConst =
        forOp.getStep().getDefiningOp<mlir::arith::ConstantIndexOp>();
    if (!stepConst) {
      return mlir::failure();
}

    auto affineFor = mlir::affine::AffineForOp::create(
        rewriter, loc, lbOperands, lbMap, ubOperands, ubMap, stepConst.value(),
        forOp.getInitArgs(),
        [](mlir::OpBuilder &, mlir::Location, mlir::Value, mlir::ValueRange) -> void {
        });

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
      if (auto loadOp = mlir::dyn_cast<mlir::memref::LoadOp>(op)) {
        llvm::SmallVector<mlir::Value> idxs;
        bool ok = true;
        for (mlir::Value idx : loadOp.getIndices()) {
          mlir::Value r = resolveIndexCastChain(mapping.lookupOrDefault(idx));
          if (!isLegalAffineIndex(r)) {
            ok = false;
            break;
          }
          idxs.push_back(r);
        }
        if (ok) {
          auto nl = mlir::affine::AffineLoadOp::create(
              rewriter, loadOp.getLoc(), loadOp.getMemRef(), idxs);
          mapping.map(loadOp.getResult(), nl.getResult());
          continue;
        }
      }
      if (auto storeOp = mlir::dyn_cast<mlir::memref::StoreOp>(op)) {
        llvm::SmallVector<mlir::Value> idxs;
        bool ok = true;
        for (mlir::Value idx : storeOp.getIndices()) {
          mlir::Value r = resolveIndexCastChain(mapping.lookupOrDefault(idx));
          if (!isLegalAffineIndex(r)) {
            ok = false;
            break;
          }
          idxs.push_back(r);
        }
        if (ok) {
          mlir::affine::AffineStoreOp::create(
              rewriter, storeOp.getLoc(),
              mapping.lookupOrDefault(storeOp.getValue()), storeOp.getMemRef(),
              idxs);
          continue;
        }
      }
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
