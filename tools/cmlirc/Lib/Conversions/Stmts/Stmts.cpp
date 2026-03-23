#include "../../Converter.h"

namespace cmlirc {

bool CMLIRConverter::TraverseStmt(clang::Stmt *stmt) {
  if (!stmt || !currentFunc)
    return RecursiveASTVisitor::TraverseStmt(stmt);

  mlir::OpBuilder &builder = contextManager.Builder();
  mlir::Block *cur = builder.getInsertionBlock();
  if (cur && !cur->empty() &&
      cur->back().hasTrait<mlir::OpTrait::IsTerminator>())
    return true;

  if (llvm::isa<clang::BreakStmt>(stmt))
    return TraverseBreakStmt(llvm::cast<clang::BreakStmt>(stmt));

  if (auto *expr = llvm::dyn_cast<clang::Expr>(stmt)) {
    if (hasSideEffects(expr)) {
      generateExpr(expr);
      return true;
    }
  }

  return RecursiveASTVisitor::TraverseStmt(stmt);
}

} // namespace cmlirc
