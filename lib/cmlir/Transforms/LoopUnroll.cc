#include "cmlir/Transforms/Passes.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Utils/Utils.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/LoopInvariantCodeMotionUtils.h"

#define GEN_PASS_DEF_LOOPUNROLLPASS
#include "cmlir/Transforms/Passes.h.inc"

namespace cmlir {

struct LoopUnrollPass : public impl::LoopUnrollPassBase<LoopUnrollPass> {

  void runOnOperation() override {
    auto op = getOperation();

    llvm::SmallVector<mlir::scf::ForOp, 8> loops;
    op->walk([&](mlir::scf::ForOp forOp) { loops.push_back(forOp); });

    for (mlir::scf::ForOp forOp : loops) {
      // #pragma cmlir loop unroll(disable)
      if (forOp->hasAttr("nounroll"))
        continue;

      // #pragma cmlir loop unroll(full)
      if (forOp->hasAttr("unroll")) {
        if (mlir::failed(mlir::loopUnrollFull(forOp))) {
          forOp->emitWarning("cmlir: full loop unroll failed");
        }
        continue;
      }

      // #pragma cmlir loop unroll(N)
      if (auto countAttr =
              forOp->getAttrOfType<mlir::IntegerAttr>("unroll_count")) {
        uint32_t factor = (uint32_t)countAttr.getInt();
        if (factor < 2)
          continue;

        if (mlir::failed(mlir::loopUnrollByFactor(forOp, factor))) {
          forOp->emitWarning("cmlir: loop unroll by factor " +
                             llvm::Twine(factor) + " failed");
        }
      }
    }
  }
};

std::unique_ptr<mlir::Pass> createLoopUnrollPass() {
  return std::make_unique<LoopUnrollPass>();
}

} // namespace cmlir
