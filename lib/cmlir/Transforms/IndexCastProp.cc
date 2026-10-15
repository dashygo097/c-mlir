#include "CastFolding.h"
#include "mlir/Analysis/DataFlow/IntegerRangeAnalysis.h"
#include "mlir/Analysis/DataFlow/Utils.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/DataLayoutInterfaces.h"
#include <type_traits>

namespace cmlir {

namespace {

template <typename Cast> auto roundTripSource(Cast cast) -> mlir::Value {
  auto inner = cast.getIn().template getDefiningOp<Cast>();
  if (!inner || inner.getIn().getType() != cast.getType() ||
      !cast.getType().isIntOrIndex()) {
    return {};
  }
  return inner.getIn();
}

template <typename Cast>
auto isExact(Cast cast, mlir::DataFlowSolver &solver) -> bool {
  auto source = roundTripSource(cast);
  if (!source) {
    return false;
  }
  auto layout = mlir::DataLayout::closest(cast);
  auto sourceWidth = layout.getTypeSizeInBits(source.getType()).getFixedValue();
  auto intermediateWidth =
      layout.getTypeSizeInBits(cast.getIn().getType()).getFixedValue();
  if (intermediateWidth >= sourceWidth) {
    return true;
  }
  auto *state =
      solver.lookupState<mlir::dataflow::IntegerValueRangeLattice>(source);
  if (!state || state->getValue().isUninitialized()) {
    return false;
  }
  const auto &range = state->getValue().getValue();
  // Index range storage is normally 64 bits. Do not use that range to prove
  // anything about a target with a wider index representation.
  if (range.smin().getBitWidth() < sourceWidth) {
    return false;
  }
  if constexpr (std::is_same_v<Cast, mlir::arith::IndexCastOp>) {
    return range.smin().isSignedIntN(intermediateWidth) &&
           range.smax().isSignedIntN(intermediateWidth);
  } else {
    return range.umin().isIntN(intermediateWidth) &&
           range.umax().isIntN(intermediateWidth);
  }
}

} // namespace

auto foldExactIndexRoundTrips(mlir::Operation *operation)
    -> mlir::LogicalResult {
  llvm::SmallVector<mlir::Operation *, 8> candidates;
  operation->walk([&](mlir::Operation *op) -> void {
    if (auto cast = mlir::dyn_cast<mlir::arith::IndexCastOp>(op)) {
      if (roundTripSource(cast)) {
        candidates.push_back(op);
      }
    } else if (auto cast = mlir::dyn_cast<mlir::arith::IndexCastUIOp>(op)) {
      if (roundTripSource(cast)) {
        candidates.push_back(op);
      }
    }
  });
  if (candidates.empty()) {
    return mlir::success();
  }
  mlir::DataFlowSolver solver;
  mlir::dataflow::loadBaselineAnalyses(solver);
  solver.load<mlir::dataflow::IntegerRangeAnalysis>();
  if (mlir::failed(solver.initializeAndRun(operation))) {
    return mlir::failure();
  }

  // Finish all range queries before mutating IR. Greedy folding and region
  // simplification could otherwise invalidate the solver's value identities.
  llvm::SmallVector<mlir::Operation *, 8> exact;
  for (auto *op : candidates) {
    bool proven = false;
    if (auto cast = mlir::dyn_cast<mlir::arith::IndexCastOp>(op)) {
      proven = isExact(cast, solver);
    } else {
      proven = isExact(mlir::cast<mlir::arith::IndexCastUIOp>(op), solver);
    }
    if (proven) {
      exact.push_back(op);
    }
  }
  mlir::IRRewriter rewriter(operation->getContext());
  for (auto *op : exact) {
    // Read the current source: another proven replacement may have shortened
    // a chain since the analysis was computed.
    mlir::Value source;
    if (auto cast = mlir::dyn_cast<mlir::arith::IndexCastOp>(op)) {
      source = roundTripSource(cast);
    } else {
      source = roundTripSource(mlir::cast<mlir::arith::IndexCastUIOp>(op));
    }
    if (source) {
      rewriter.replaceOp(op, source);
    }
  }
  return mlir::success();
}

} // namespace cmlir
