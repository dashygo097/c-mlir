#include "cmlir/Transforms/Passes.h"
#include "mlir/Pass/Pass.h"

namespace cmlir {

#define GEN_PASS_DEF_RAISEMEMREF2AFFINEPASS
#include "cmlir/Transforms/Passes.h.inc"

struct RaiseMemref2AffinePass
    : public impl::RaiseMemref2AffinePassBase<RaiseMemref2AffinePass> {
  void runOnOperation() override {}
};

std::unique_ptr<mlir::Pass> createRaiseMemref2AffinePass() {
  return std::make_unique<RaiseMemref2AffinePass>();
}

} // namespace cmlir
