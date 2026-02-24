#include "cmlir/Transforms/Passes.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Utils/Utils.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/LoopInvariantCodeMotionUtils.h"

namespace cmlir {

struct LoopUnrollPass
    : public mlir::PassWrapper<LoopUnrollPass,
                               mlir::OperationPass<mlir::func::FuncOp>> {

  void runOnOperation() override {
    mlir::func::FuncOp func = getOperation();

    llvm::SmallVector<mlir::scf::ForOp, 8> loops;
    func.walk([&](mlir::scf::ForOp forOp) { loops.push_back(forOp); });

    for (mlir::scf::ForOp forOp : loops) {
      // #pragma cmlir loop_unroll(disable)
      if (forOp->hasAttr("nounroll"))
        continue;

      // #pragma cmlir loop_unroll(full)
      if (forOp->hasAttr("unroll")) {
        if (mlir::failed(mlir::loopUnrollFull(forOp))) {
          forOp->emitWarning("cmlir: full loop unroll failed");
        }
        continue;
      }

      // #pragma cmlir loop_unroll(N)
      if (auto countAttr =
              forOp->getAttrOfType<mlir::IntegerAttr>("unroll_count")) {
        uint64_t factor = (uint64_t)countAttr.getInt();
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
