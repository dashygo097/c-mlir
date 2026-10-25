#include "../ExpressionLowering.h"

namespace cmlir::frontend {

auto ExpressionLowering::generateBinaryOperator(clang::BinaryOperator *binOp)
    -> mlir::Value {
  if (binOp->isAssignmentOp()) {
    return generateAssignmentBinaryOperator(binOp);
  }
  return generatePureBinaryOperator(binOp);
}

} // namespace cmlir::frontend
