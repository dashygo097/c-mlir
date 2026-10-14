#include "../Lowering.h"
#include "clang-mlir/Casts.h"
#include "mlir/Dialect/SCF/IR/SCF.h"

namespace cmlir::frontend {

auto ASTLowering::generateConditionalOperator(
    clang::ConditionalOperator *condOp) -> mlir::Value {
  mlir::OpBuilder &builder = context.builder();
  mlir::Location loc = context.source().location(condOp->getExprLoc());

  mlir::Value condition = generateExpr(condOp->getCond());
  mlir::Value condBool = clang_mlir::toBool(builder, loc, condition);

  mlir::Type targetType = convertType(condOp->getType());

  auto ifOp = mlir::scf::IfOp::create(builder, loc, targetType, condBool,
                                      /*hasElse=*/true);

  {
    mlir::OpBuilder::InsertionGuard guard(builder);
    mlir::Block *thenBlock = &ifOp.getThenRegion().front();
    builder.setInsertionPointToStart(thenBlock);

    auto *trueExpr = condOp->getTrueExpr();
    mlir::Value trueValue =
        trueExpr->isGLValue() ? loadLValue(trueExpr) : generateExpr(trueExpr);
    if (!trueValue) {
      return {};
    }
    trueValue = clang_mlir::toValue(builder, loc, trueValue, targetType, true);

    mlir::scf::YieldOp::create(builder, loc, trueValue);
  }

  {
    mlir::OpBuilder::InsertionGuard guard(builder);
    mlir::Block *elseBlock = &ifOp.getElseRegion().front();
    builder.setInsertionPointToStart(elseBlock);

    mlir::Value falseValue;
    if (condOp->getFalseExpr()) {
      auto *falseExpr = condOp->getFalseExpr();
      falseValue = falseExpr->isGLValue() ? loadLValue(falseExpr)
                                          : generateExpr(falseExpr);
      if (!falseValue) {
        return {};
      }
      falseValue =
          clang_mlir::toValue(builder, loc, falseValue, targetType, true);
    } else {
      falseValue = clang_mlir::intConst(builder, loc, targetType, 0);
    }

    mlir::scf::YieldOp::create(builder, loc, falseValue);
  }

  builder.setInsertionPointAfter(ifOp);
  return ifOp.getResult(0);
}

} // namespace cmlir::frontend
