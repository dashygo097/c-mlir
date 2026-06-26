#include "../../../Converter.h"

namespace cmlirc {

auto CMLIRConverter::generateBinaryOperator(clang::BinaryOperator *binOp)
    -> mlir::Value {
  if (binOp->isAssignmentOp()) {
    return generateAssignmentBinaryOperator(binOp);
  }
  return generatePureBinaryOperator(binOp);
}

} // namespace cmlirc
