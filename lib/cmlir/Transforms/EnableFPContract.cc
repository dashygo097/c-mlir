#include "cmlir/Transforms/Passes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"

#define GEN_PASS_DEF_ENABLEFPCONTRACTPASS
#include "cmlir/Transforms/Passes.h.inc"

namespace cmlir {

namespace {

struct EnableFPContractPass
    : impl::EnableFPContractPassBase<EnableFPContractPass> {
  auto runOnOperation() -> void override {
    getOperation()->walk([](mlir::Operation *op) -> void {
      if (auto add = mlir::dyn_cast<mlir::arith::AddFOp>(op)) {
        add.setFastmath(add.getFastmath() |
                        mlir::arith::FastMathFlags::contract);
      } else if (auto mul = mlir::dyn_cast<mlir::arith::MulFOp>(op)) {
        mul.setFastmath(mul.getFastmath() |
                        mlir::arith::FastMathFlags::contract);
      }
    });
  }
};

} // namespace

auto createEnableFPContractPass() -> std::unique_ptr<mlir::Pass> {
  return std::make_unique<EnableFPContractPass>();
}

} // namespace cmlir
