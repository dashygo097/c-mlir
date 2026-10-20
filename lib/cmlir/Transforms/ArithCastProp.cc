#include "CastFolding.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/APFloat.h"

#define GEN_PASS_DEF_ARITHCASTPROPPASS
#include "cmlir/Transforms/Passes.h.inc"

namespace cmlir {

namespace {

// Eliminate a round trip only when every source integer is exactly
// representable. Moving arithmetic across casts also needs a proof about
// intermediate results, so those former rewrites are deliberately absent.
template <typename ToInteger, typename ToFloat, bool Signed>
struct ExactIntegerRoundTrip : mlir::OpRewritePattern<ToInteger> {
  using mlir::OpRewritePattern<ToInteger>::OpRewritePattern;

  auto matchAndRewrite(ToInteger cast, mlir::PatternRewriter &rewriter) const
      -> mlir::LogicalResult override {
    auto conversion = cast.getIn().template getDefiningOp<ToFloat>();
    if (!conversion || conversion.getIn().getType() != cast.getType()) {
      return mlir::failure();
    }
    auto integer = mlir::dyn_cast<mlir::IntegerType>(cast.getType());
    auto floating = mlir::dyn_cast<mlir::FloatType>(conversion.getType());
    if (!integer || !floating ||
        !(floating.isF16() || floating.isBF16() || floating.isF32() ||
          floating.isF64() || floating.isF80() || floating.isF128())) {
      return mlir::failure();
    }
    const auto &semantics = floating.getFloatSemantics();
    auto valueBits = integer.getWidth() - (Signed ? 1 : 0);
    if (valueBits > llvm::APFloat::semanticsPrecision(semantics) ||
        integer.getWidth() - 1 >
            static_cast<unsigned>(
                llvm::APFloat::semanticsMaxExponent(semantics))) {
      return mlir::failure();
    }
    rewriter.replaceOp(cast, conversion.getIn());
    return mlir::success();
  }
};

} // namespace

struct ArithCastPropPass
    : public impl::ArithCastPropPassBase<ArithCastPropPass> {
  auto runOnOperation() -> void override {
    if (mlir::failed(foldExactIndexRoundTrips(getOperation()))) {
      signalPassFailure();
      return;
    }
    mlir::RewritePatternSet patterns(&getContext());
    patterns.add<ExactIntegerRoundTrip<mlir::arith::FPToSIOp,
                                       mlir::arith::SIToFPOp, true>,
                 ExactIntegerRoundTrip<mlir::arith::FPToUIOp,
                                       mlir::arith::UIToFPOp, false>>(
        &getContext());
    // Index round trips were proved against the target data layout above.
    // Generic op folding can assume a 64-bit index and bypass that proof.
    mlir::GreedyRewriteConfig config;
    config.enableFolding(false);
    if (mlir::failed(mlir::applyPatternsGreedily(
            getOperation(), std::move(patterns), config))) {
      signalPassFailure();
    }
  }
};

auto createArithCastPropPass() -> std::unique_ptr<mlir::Pass> {
  return std::make_unique<ArithCastPropPass>();
}

} // namespace cmlir
