#include "../Lowering.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"

namespace cmlir::frontend {

auto ASTLowering::emitBreak(clang::BreakStmt *stmt) -> bool {
  for (auto it = function.controls.rbegin(); it != function.controls.rend();
       ++it) {
    if (!it->exitBlock) {
      continue;
    }
    auto &builder = context.builder();
    mlir::cf::BranchOp::create(
        builder, context.source().location(stmt->getBeginLoc()), it->exitBlock);
    return true;
  }
  context.source().error(stmt->getBeginLoc(),
                         "missing control-flow destination");
  return false;
}

} // namespace cmlir::frontend
