#include "../Converter.h"

namespace cmlirc {
bool CMLIRConverter::hasSideEffects(clang::Expr *expr) const {
  if (auto *unOp = llvm::dyn_cast<clang::UnaryOperator>(expr)) {
    return unOp->isIncrementDecrementOp();
  }

  if (auto *binOp = llvm::dyn_cast<clang::BinaryOperator>(expr)) {
    return binOp->isAssignmentOp() || binOp->isCompoundAssignmentOp();
  }

  if (llvm::isa<clang::CallExpr>(expr)) {
    return true;
  }

  return false;
}

bool CMLIRConverter::branchEndsWithReturn(clang::Stmt *stmt) {
  if (!stmt)
    return false;

  if (llvm::isa<clang::ReturnStmt>(stmt)) {
    return true;
  }

  if (auto *compound = llvm::dyn_cast<clang::CompoundStmt>(stmt)) {
    if (compound->body_empty())
      return false;
    return branchEndsWithReturn(compound->body_back());
  }

  if (auto *ifStmt = llvm::dyn_cast<clang::IfStmt>(stmt)) {
    return branchEndsWithReturn(ifStmt->getThen()) &&
           (ifStmt->getElse() ? branchEndsWithReturn(ifStmt->getElse())
                              : false);
  }

  return false;
}

} // namespace cmlirc
