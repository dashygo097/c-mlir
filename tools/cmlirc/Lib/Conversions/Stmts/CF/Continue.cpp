#include "../../../Converter.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/IR/Block.h"

namespace cmlirc {
bool CMLIRConverter::TraverseContinueStmt(clang::ContinueStmt *) {
  if (!currentFunc || loopStack.empty())
    return true;

  mlir::OpBuilder &builder = context_manager_.Builder();
  mlir::Location loc = builder.getUnknownLoc();

  mlir::Block *target = loopStack.back().headerBlock;
  if (!target) {
    llvm::errs() << "continue: no header block\n";
    return false;
  }

  mlir::cf::BranchOp::create(builder, loc, target, mlir::ValueRange{});

  mlir::Region *region = builder.getInsertionBlock()->getParent();
  mlir::Block *dead = builder.createBlock(region);
  builder.setInsertionPointToStart(dead);
  return true;
}

} // namespace cmlirc
