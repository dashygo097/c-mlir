#include "../../../Converter.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"

namespace cmlirc {
bool CMLIRConverter::TraverseBreakStmt(clang::BreakStmt *) {
  if (!currentFunc || loopStack.empty())
    return true;

  mlir::OpBuilder &builder = context_manager_.Builder();
  mlir::Location loc = builder.getUnknownLoc();

  mlir::Block *target = loopStack.back().exitBlock;
  if (!target) {
    llvm::errs() << "break: no exit block\n";
    return false;
  }

  mlir::cf::BranchOp::create(builder, loc, target, mlir::ValueRange{});

  mlir::Region *region = builder.getInsertionBlock()->getParent();
  mlir::Block *dead = builder.createBlock(region);
  builder.setInsertionPointToStart(dead);
  return true;
}

} // namespace cmlirc
