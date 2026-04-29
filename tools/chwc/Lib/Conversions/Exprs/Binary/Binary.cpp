#include "../../../Converter.h"

namespace chwc {

auto CHWConverter::generateBinaryOperator(clang::BinaryOperator *binOp)
    -> mlir::Value {
  if (binOp->isAssignmentOp()) {
    return generateAssignmentBinaryOperator(binOp);
  }

  return generatePureBinaryOperator(binOp);
}

} // namespace chwc
