#include "../../../Converter.h"

namespace cmlirc {

mlir::Value
CMLIRConverter::generateBinaryOperator(clang::BinaryOperator *binOp) {
  if (binOp->isAssignmentOp())
    return generateAssignmentBinaryOperator(binOp);
  return generatePureBinaryOperator(binOp);
}

} // namespace cmlirc
