#include "cmlir/Transforms/Passes.h"
#include "mlir/Pass/Pass.h"

namespace cmlir {

#define GEN_PASS_DEF_RAISE2AFFINEPASS
#include "cmlir/Transforms/Passes.h.inc"

struct Raise2AffinePass : public impl::Raise2AffinePassBase<Raise2AffinePass> {
  void runOnOperation() override {}
};

std::unique_ptr<mlir::Pass> createRaise2AffinePass() {
  return std::make_unique<Raise2AffinePass>();
}

} // namespace cmlir
