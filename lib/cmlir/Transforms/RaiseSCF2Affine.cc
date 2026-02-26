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

static mlir::Value resolveIndexCastChain(mlir::Value v) {
  auto outerCast = v.getDefiningOp<mlir::arith::IndexCastOp>();
  if (!outerCast)
    return v;

  mlir::Value mid = outerCast.getIn();
  if (mid.getType().isIndex())
    return resolveIndexCastChain(mid);

  auto innerCast = mid.getDefiningOp<mlir::arith::IndexCastOp>();
  if (!innerCast)
    return v;

  mlir::Value original = innerCast.getIn();
  if (!original.getType().isIndex())
    return v;

  return resolveIndexCastChain(original);
}

static bool isLegalAffineIndex(mlir::Value v) {
  if (mlir::affine::isValidDim(v) || mlir::affine::isValidSymbol(v))
    return true;

  if (auto blockArg = llvm::dyn_cast<mlir::BlockArgument>(v)) {
    mlir::Operation *parentOp = blockArg.getOwner()->getParentOp();
    if (auto forOp = llvm::dyn_cast_or_null<mlir::scf::ForOp>(parentOp))
      if (blockArg == forOp.getInductionVar())
        return true;
  }

  return false;
}

struct RaiseSCFFor2AffineForPattern
    : public mlir::OpRewritePattern<mlir::scf::ForOp> {
  using mlir::OpRewritePattern<mlir::scf::ForOp>::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::scf::ForOp forOp,
                  mlir::PatternRewriter &rewriter) const override {

    auto lbConst =
        forOp.getLowerBound().getDefiningOp<mlir::arith::ConstantIndexOp>();
    auto ubConst =
        forOp.getUpperBound().getDefiningOp<mlir::arith::ConstantIndexOp>();
    auto stepConst =
        forOp.getStep().getDefiningOp<mlir::arith::ConstantIndexOp>();
    if (!lbConst || !ubConst || !stepConst)
      return mlir::failure();

    mlir::Location loc = forOp.getLoc();

    auto affineFor = mlir::affine::AffineForOp::create(
        rewriter, loc, lbConst.value(), ubConst.value(), stepConst.value(),
        forOp.getInitArgs(),
        [](mlir::OpBuilder &, mlir::Location, mlir::Value, mlir::ValueRange) {
        });

    mlir::Block *affineBody = affineFor.getBody();
    mlir::Block *scfBody = forOp.getBody();

    if (!affineBody->empty() &&
        affineBody->back().hasTrait<mlir::OpTrait::IsTerminator>())
      rewriter.eraseOp(&affineBody->back());

    mlir::IRMapping mapping;
    mapping.map(forOp.getInductionVar(), affineFor.getInductionVar());
    for (auto [scfArg, affineArg] :
         llvm::zip(forOp.getRegionIterArgs(), affineFor.getRegionIterArgs()))
      mapping.map(scfArg, affineArg);

    rewriter.setInsertionPointToEnd(affineBody);
    for (auto &op : scfBody->without_terminator()) {

      // memref.load → affine.load
      if (auto loadOp = mlir::dyn_cast<mlir::memref::LoadOp>(op)) {
        llvm::SmallVector<mlir::Value> fullIndices;
        bool legalIndices = true;
        for (mlir::Value index : loadOp.getIndices()) {
          mlir::Value resolved =
              resolveIndexCastChain(mapping.lookupOrDefault(index));
          if (!isLegalAffineIndex(resolved)) {
            legalIndices = false;
            break;
          }
          fullIndices.push_back(resolved);
        }
        if (legalIndices) {
          auto newLoad = mlir::affine::AffineLoadOp::create(
              rewriter, loadOp.getLoc(), loadOp.getMemRef(), fullIndices);
          mapping.map(loadOp.getResult(), newLoad.getResult());
          continue;
        }
      }

      // memref.store → affine.store
      if (auto storeOp = mlir::dyn_cast<mlir::memref::StoreOp>(op)) {
        llvm::SmallVector<mlir::Value> fullIndices;
        bool legalIndices = true;
        for (mlir::Value index : storeOp.getIndices()) {
          mlir::Value resolved =
              resolveIndexCastChain(mapping.lookupOrDefault(index));
          if (!isLegalAffineIndex(resolved)) {
            legalIndices = false;
            break;
          }
          fullIndices.push_back(resolved);
        }
        if (legalIndices) {
          mlir::affine::AffineStoreOp::create(
              rewriter, storeOp.getLoc(),
              mapping.lookupOrDefault(storeOp.getValue()), storeOp.getMemRef(),
              fullIndices);
          continue;
        }
      }

      // Generic fallback: clone with mapping
      rewriter.clone(op, mapping);
    }

    auto scfYield = mlir::cast<mlir::scf::YieldOp>(scfBody->getTerminator());
    llvm::SmallVector<mlir::Value> yieldOperands;
    for (mlir::Value v : scfYield.getOperands())
      yieldOperands.push_back(mapping.lookupOrDefault(v));
    mlir::affine::AffineYieldOp::create(rewriter, scfYield.getLoc(),
                                        yieldOperands);

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
