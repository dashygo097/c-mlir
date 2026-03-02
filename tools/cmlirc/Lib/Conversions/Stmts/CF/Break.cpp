#include "../../../Converter.h"
#include "mlir/Dialect/SCF/IR/SCF.h"

namespace cmlirc {

bool CMLIRConverter::TraverseBreakStmt(clang::BreakStmt *) {
  if (!currentFunc)
    return true;

  mlir::OpBuilder &builder = context_manager_.Builder();
  mlir::Location loc = builder.getUnknownLoc();

  if (breakStack.empty()) {
    llvm::errs() << "error: break outside switch — not supported in scf "
                    "dialect.\n";
    return false;
  }

  auto &top = breakStack.back();
  if (top.kind != BreakTargetKind::ScfYield) {
    llvm::errs() << "error: break in loop context — not supported in pure "
                    "scf dialect.\n";
    return false;
  }

  mlir::scf::YieldOp::create(builder, loc, mlir::ValueRange{});
  return true;
}

} // namespace cmlirc
