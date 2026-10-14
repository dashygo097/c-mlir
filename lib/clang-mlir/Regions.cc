#include "clang-mlir/Regions.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Diagnostics.h"

namespace clang_mlir {

auto hasTerminator(mlir::Block *block) -> bool {
  return block && !block->empty() &&
         block->back().hasTrait<mlir::OpTrait::IsTerminator>();
}

auto emitArm(mlir::OpBuilder &builder, mlir::Location loc, mlir::Block &arm,
             mlir::Block *continuation,
             llvm::function_ref<mlir::LogicalResult()> emitBody)
    -> mlir::LogicalResult {
  mlir::OpBuilder::InsertionGuard guard(builder);
  if (continuation && continuation->getParent() != arm.getParent()) {
    mlir::emitError(loc, "branch continuation belongs to another region");
    return mlir::failure();
  }
  if (!arm.empty() && mlir::isa<mlir::scf::YieldOp>(arm.back())) {
    arm.back().erase();
  }
  builder.setInsertionPointToStart(&arm);
  if (mlir::failed(emitBody())) {
    return mlir::failure();
  }
  auto *exit = builder.getInsertionBlock();
  if (!exit || exit->getParent() != arm.getParent()) {
    mlir::emitError(loc, "lowered arm escaped its region");
    return mlir::failure();
  }
  if (!hasTerminator(exit)) {
    builder.setInsertionPointToEnd(exit);
    if (continuation) {
      mlir::cf::BranchOp::create(builder, loc, continuation);
    } else {
      mlir::scf::YieldOp::create(builder, loc);
    }
  }
  return mlir::success();
}

} // namespace clang_mlir
