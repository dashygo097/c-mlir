#include "../../Converter.h"

namespace cmlirc {

bool CMLIRConverter::hasSideEffects(clang::Expr *expr) const {
  if (auto *unOp = mlir::dyn_cast<clang::UnaryOperator>(expr)) {
    return unOp->isIncrementDecrementOp();
  }

  if (auto *binOp = mlir::dyn_cast<clang::BinaryOperator>(expr)) {
    return binOp->isAssignmentOp() || binOp->isCompoundAssignmentOp();
  }

  if (mlir::isa<clang::CallExpr>(expr)) {
    return true;
  }

  return false;
}

} // namespace cmlirc
