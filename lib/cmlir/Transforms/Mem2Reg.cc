#include "cmlir/Transforms/Passes.h"
#include "mlir/Pass/Pass.h"

namespace cmlir {

#define GEN_PASS_DEF_MEM2REGPASS
#include "cmlir/Transforms/Passes.h.inc"

struct Mem2RegPass : public impl::Mem2RegPassBase<Mem2RegPass> {
  void runOnOperation() override {}
};

std::unique_ptr<mlir::Pass> createMem2RegPass() {
  return std::make_unique<Mem2RegPass>();
}

} // namespace cmlir
