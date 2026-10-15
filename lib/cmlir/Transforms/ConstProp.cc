#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#define GEN_PASS_DEF_CONSTPROPPASS
#include "cmlir/Transforms/Passes.h.inc"

namespace cmlir {

struct ConstPropPass : public impl::ConstPropPassBase<ConstPropPass> {
  auto runOnOperation() -> void override {
    // Operation folders preserve APInt/APFloat widths, signedness and rounding.
    // Host double/int64_t conversions cannot safely implement these casts.
    mlir::RewritePatternSet patterns(&getContext());
    if (mlir::failed(
            mlir::applyPatternsGreedily(getOperation(), std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

auto createConstPropPass() -> std::unique_ptr<mlir::Pass> {
  return std::make_unique<ConstPropPass>();
}

} // namespace cmlir
