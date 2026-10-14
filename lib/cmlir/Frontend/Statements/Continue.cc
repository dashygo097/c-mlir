#include "../Lowering.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"

namespace cmlir::frontend {

auto ASTLowering::emitContinue(clang::ContinueStmt *stmt) -> bool {
  for (auto it = function.controls.rbegin(); it != function.controls.rend();
       ++it) {
    if (!it->continueBlock) {
      continue;
    }
    auto &builder = context.builder();
    mlir::cf::BranchOp::create(builder,
                               context.source().location(stmt->getBeginLoc()),
                               it->continueBlock);
    return true;
  }
  context.source().error(stmt->getBeginLoc(),
                         "missing control-flow destination");
  return false;
}

} // namespace cmlir::frontend
