#include "../../Lowering.h"

namespace cmlir::frontend {

auto ASTLowering::generateBinaryOperator(clang::BinaryOperator *binOp)
    -> mlir::Value {
  if (binOp->isAssignmentOp()) {
    return generateAssignmentBinaryOperator(binOp);
  }
  return generatePureBinaryOperator(binOp);
}

} // namespace cmlir::frontend
