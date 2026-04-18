#include "../../Converter.h"
#include "../Utils/Casts.h"
#include "mlir/Dialect/SCF/IR/SCF.h"

namespace cmlirc {

auto CMLIRConverter::generateConditionalOperator(
    clang::ConditionalOperator *condOp) -> mlir::Value {
  mlir::OpBuilder &builder = contextManager.Builder();
  mlir::Location loc = builder.getUnknownLoc();

  mlir::Value condition = generateExpr(condOp->getCond());
  if (!condition) {
    return nullptr;
  }
  mlir::Value condBool = utils::toBool(builder, loc, condition);

  mlir::Type targetType = convertType(condOp->getType());

  auto ifOp = mlir::scf::IfOp::create(builder, loc, targetType, condBool,
                                      /*hasElse=*/true);

  {
    mlir::OpBuilder::InsertionGuard guard(builder);
    mlir::Block *thenBlock = &ifOp.getThenRegion().front();
    builder.setInsertionPointToStart(thenBlock);

    mlir::Value trueValue = generateExpr(condOp->getTrueExpr());
    trueValue = utils::castValue(builder, loc, trueValue, targetType, true);

    mlir::scf::YieldOp::create(builder, loc, trueValue);
  }

  {
    mlir::OpBuilder::InsertionGuard guard(builder);
    mlir::Block *elseBlock = &ifOp.getElseRegion().front();
    builder.setInsertionPointToStart(elseBlock);

    mlir::Value falseValue;
    if (condOp->getFalseExpr()) {
      falseValue = generateExpr(condOp->getFalseExpr());
      falseValue = utils::castValue(builder, loc, falseValue, targetType, true);
    } else {
      falseValue = utils::intConst(builder, loc, targetType, 0);
    }

    mlir::scf::YieldOp::create(builder, loc, falseValue);
  }

  builder.setInsertionPointAfter(ifOp);
  return ifOp.getResult(0);
}

} // namespace cmlirc
