#include "../../Converter.h"
#include "../Utils/Constants.h"

namespace cmlirc {

auto CMLIRConverter::generateConditionalOperator(
    clang::ConditionalOperator *condOp) -> mlir::Value {
  mlir::OpBuilder &builder = contextManager.Builder();
  mlir::Location loc = builder.getUnknownLoc();

  mlir::Value condition = generateExpr(condOp->getCond());
  mlir::Value trueValue = generateExpr(condOp->getTrueExpr());
  mlir::Value falseValue;
  mlir::Type resultType = trueValue.getType();

  falseValue = condOp->getFalseExpr()
                   ? generateExpr(condOp->getFalseExpr())
                   : utils::intConst(builder, loc, resultType, 0);

  mlir::Value result = mlir::arith::SelectOp::create(builder, loc, condition,
                                                     trueValue, falseValue);

  return result;
}
} // namespace cmlirc
