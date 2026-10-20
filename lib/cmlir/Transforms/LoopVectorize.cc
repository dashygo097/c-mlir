#include "cmlir/Transforms/Passes.h"
#include "mlir/Analysis/AliasAnalysis.h"
#include "mlir/Conversion/AffineToStandard/AffineToStandard.h"
#include "mlir/Conversion/SCFToAffine/SCFToAffine.h"
#include "mlir/Dialect/Affine/Analysis/AffineAnalysis.h"
#include "mlir/Dialect/Affine/LoopUtils.h"
#include "mlir/Dialect/Affine/Transforms/Passes.h"
#include "mlir/Dialect/Affine/Utils.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Interfaces/ViewLikeInterface.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Rewrite/PatternApplicator.h"
#include "mlir/Transforms/Passes.h"
#include <limits>

#define GEN_PASS_DEF_LOOPVECTORIZEPASS
#include "cmlir/Transforms/Passes.h.inc"

namespace cmlir {

namespace {

// Observe the replacement without depending on the upstream pattern's final
// insertion point. SCF raising creates one new Affine loop and moves its body.
struct RaisedLoop : mlir::OpBuilder::Listener {
  mlir::affine::AffineForOp loop;

  auto notifyOperationInserted(mlir::Operation *op,
                               mlir::OpBuilder::InsertPoint previous)
      -> void override {
    if (!previous.isSet() && !loop) {
      loop = mlir::dyn_cast<mlir::affine::AffineForOp>(op);
    }
  }
};

auto allocationRoot(mlir::Value value) -> mlir::Value {
  while (auto *op = value.getDefiningOp()) {
    if (auto view = mlir::dyn_cast<mlir::ViewLikeOpInterface>(op)) {
      value = view.getViewSource();
    } else if (auto cast = mlir::dyn_cast<mlir::memref::CastOp>(op)) {
      value = cast.getSource();
    } else {
      break;
    }
  }
  return value;
}

auto isRestrictArgument(mlir::Value value, mlir::func::FuncOp function)
    -> bool {
  auto arg = mlir::dyn_cast<mlir::BlockArgument>(value);
  return arg && arg.getOwner() == &function.front() &&
         function.getArgAttr(arg.getArgNumber(), "cmlir.restrict");
}

auto hasSafeAliases(mlir::affine::AffineForOp loop, mlir::func::FuncOp function)
    -> bool {
  llvm::SmallVector<mlir::Value> accesses, writes;
  loop.walk([&](mlir::Operation *op) -> void {
    if (auto read = mlir::dyn_cast<mlir::affine::AffineReadOpInterface>(op)) {
      accesses.push_back(read.getMemRef());
    } else if (auto write =
                   mlir::dyn_cast<mlir::affine::AffineWriteOpInterface>(op)) {
      accesses.push_back(write.getMemRef());
      writes.push_back(write.getMemRef());
    }
  });
  mlir::AliasAnalysis analysis(function);
  for (auto write : writes) {
    for (auto access : accesses) {
      if (write == access || analysis.alias(write, access).isNo()) {
        continue;
      }
      auto lhs = allocationRoot(write);
      auto rhs = allocationRoot(access);
      if (lhs != rhs && (isRestrictArgument(lhs, function) ||
                         isRestrictArgument(rhs, function))) {
        continue;
      }
      return false;
    }
  }
  return true;
}

auto permitsReduction(mlir::affine::AffineForOp loop) -> bool {
  auto yield =
      mlir::cast<mlir::affine::AffineYieldOp>(loop.getBody()->getTerminator());
  for (auto value : yield.getOperands()) {
    if (!mlir::isa<mlir::FloatType>(value.getType())) {
      continue;
    }
    auto op = mlir::dyn_cast_if_present<mlir::arith::ArithFastMathInterface>(
        value.getDefiningOp());
    auto flags =
        op ? op.getFastMathFlagsAttr() : mlir::arith::FastMathFlagsAttr{};
    if (!flags || !static_cast<bool>(flags.getValue() &
                                     mlir::arith::FastMathFlags::reassoc)) {
      return false;
    }
  }
  return true;
}

auto prepareFullTile(mlir::affine::AffineForOp loop, unsigned width)
    -> mlir::FailureOr<mlir::affine::AffineForOp> {
  if (loop.getStepAsInt() != 1) {
    return mlir::failure();
  }
  if (auto trips = loop.getStaticTripCount()) {
    if (trips->urem(width) == 0) {
      return loop;
    }
  }
  // Tiling utilities currently do not support loop-carried results. Keep
  // reductions scalar unless the entire trip count is a multiple of width.
  if (loop.getNumResults() != 0) {
    return mlir::failure();
  }
  llvm::SmallVector<mlir::affine::AffineForOp> band{loop}, tiled, full;
  if (mlir::failed(mlir::affine::tilePerfectlyNested(band, {width}, &tiled))) {
    return mlir::failure();
  }
  band.assign(1, tiled.back());
  if (mlir::failed(mlir::affine::separateFullTiles(band, &full))) {
    return mlir::failure();
  }
  return full.front();
}

} // namespace

struct LoopVectorizePass
    : public impl::LoopVectorizePassBase<LoopVectorizePass> {
  auto getDependentDialects(mlir::DialectRegistry &registry) const
      -> void override {
    registry.insert<mlir::affine::AffineDialect, mlir::arith::ArithDialect,
                    mlir::memref::MemRefDialect, mlir::scf::SCFDialect,
                    mlir::vector::VectorDialect>();
  }

  auto vectorizeFunction(mlir::func::FuncOp function,
                         mlir::PatternApplicator &raiser)
      -> mlir::LogicalResult {
    llvm::SmallVector<mlir::scf::ForOp> requested;
    function.walk([&](mlir::scf::ForOp loop) -> void {
      if (loop->hasAttr("vectorize")) {
        requested.push_back(loop);
      }
    });
    if (requested.empty()) {
      return mlir::success();
    }

    llvm::SmallVector<std::pair<mlir::affine::AffineForOp, int64_t>> candidates;
    for (auto loop : requested) {
      int64_t width = 4;
      if (auto attr =
              loop->getAttrOfType<mlir::IntegerAttr>("vectorize_width")) {
        width = attr.getInt();
      }
      loop->removeAttr("vectorize");
      loop->removeAttr("vectorize_width");
      if (width < 2 || width > std::numeric_limits<unsigned>::max()) {
        continue;
      }

      RaisedLoop replacement;
      mlir::PatternRewriter rewriter(&getContext());
      rewriter.setListener(&replacement);
      rewriter.setInsertionPoint(loop);
      if (mlir::failed(raiser.matchAndRewrite(loop, rewriter))) {
        loop.emitWarning(
            "cmlir: loop vectorization skipped: non-affine bounds");
        continue;
      }
      if (!replacement.loop) {
        return mlir::failure();
      }
      candidates.emplace_back(replacement.loop, width);
    }
    if (candidates.empty()) {
      return mlir::success();
    }

    mlir::OpPassManager accesses(mlir::func::FuncOp::getOperationName());
    accesses.addPass(mlir::affine::createRaiseMemrefToAffine());
    if (mlir::failed(runPipeline(accesses, function))) {
      return mlir::failure();
    }

    // Post-order collection processes inner loops before their parents. Only
    // the explicitly requested loop is passed to the upstream vectorizer.
    for (auto [loop, width] : candidates) {
      auto loc = loop.getLoc();
      llvm::SmallVector<mlir::affine::LoopReduction> reductions;
      if (!hasSafeAliases(loop, function) || !permitsReduction(loop) ||
          !mlir::affine::isLoopParallel(loop, &reductions)) {
        loop.emitWarning(
            "cmlir: loop vectorization skipped: alias, dependence or "
            "strict floating-point reduction");
        continue;
      }
      auto full = prepareFullTile(loop, static_cast<unsigned>(width));
      if (mlir::failed(full)) {
        mlir::emitWarning(loc, "cmlir: loop vectorization skipped: "
                               "cannot isolate full vectors");
        continue;
      }
      loop = *full;
      mlir::affine::VectorizationStrategy strategy;
      strategy.vectorSizes.push_back(width);
      strategy.loopToVectorDim[loop] = 0;
      if (!reductions.empty()) {
        strategy.reductionLoops[loop] = reductions;
      }
      std::vector<llvm::SmallVector<mlir::affine::AffineForOp, 2>> nest{{loop}};
      if (mlir::failed(mlir::affine::vectorizeAffineLoopNest(nest, strategy))) {
        loop.emitWarning("cmlir: loop vectorization skipped: unsupported body");
      }
    }

    // Affine is an implementation detail of pragma vectorization. Return to
    // SCF so callers retain the existing structured output dialect contract.
    mlir::OpPassManager lower(mlir::func::FuncOp::getOperationName());
    lower.addPass(mlir::createCanonicalizerPass());
    lower.addPass(mlir::createLowerAffinePass());
    return runPipeline(lower, function);
  }

  auto runOnOperation() -> void override {
    mlir::RewritePatternSet patterns(&getContext());
    mlir::populateSCFToAffineConversionPatterns(patterns);
    mlir::FrozenRewritePatternSet frozen(std::move(patterns));
    mlir::PatternApplicator raiser(frozen);
    raiser.applyDefaultCostModel();
    bool failed = false;
    getOperation()->walk([&](mlir::func::FuncOp function) -> void {
      if (!failed) {
        failed = mlir::failed(vectorizeFunction(function, raiser));
      }
    });
    if (failed) {
      signalPassFailure();
    }
  }
};

auto createLoopVectorizePass() -> std::unique_ptr<mlir::Pass> {
  return std::make_unique<LoopVectorizePass>();
}

} // namespace cmlir
