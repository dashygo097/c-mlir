#include "../../Converter.h"

namespace cmlirc {

mlir::Value CMLIRConverter::generateConditionalOperator(
    clang::ConditionalOperator *condOp) {
  mlir::OpBuilder &builder = context_manager_.Builder();
  mlir::Location loc = builder.getUnknownLoc();

  mlir::Value condition = generateExpr(condOp->getCond());
  mlir::Value trueValue = generateExpr(condOp->getTrueExpr());
  mlir::Value falseValue;
  mlir::Type resultType = trueValue.getType();

  if (condOp->getFalseExpr()) {
    falseValue = generateExpr(condOp->getFalseExpr());
  } else {
    falseValue = mlir::arith::ConstantOp::create(
        builder, loc, builder.getIntegerType(32),
        builder.getIntegerAttr(resultType, 0));
  }

  mlir::Value result = mlir::arith::SelectOp::create(builder, loc, condition,
                                                     trueValue, falseValue);

  return result;
}
} // namespace cmlirc
