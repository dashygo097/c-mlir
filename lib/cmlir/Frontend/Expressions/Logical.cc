#include "../Lowering.h"
#include "clang-mlir/Casts.h"
#include "mlir/Dialect/SCF/IR/SCF.h"

namespace cmlir::frontend {

auto ASTLowering::generateLogicalOperator(clang::BinaryOperator *expr)
    -> mlir::Value {
  auto &builder = context.builder();
  auto loc = context.source().location(expr->getOperatorLoc());
  auto lhs = clang_mlir::toBool(builder, loc, generateExpr(expr->getLHS()));
  if (!lhs) {
    return {};
  }
  bool isAnd = expr->getOpcode() == clang::BO_LAnd;
  auto branch =
      mlir::scf::IfOp::create(builder, loc, builder.getI1Type(), lhs, true);
  auto *rhsBlock = isAnd ? branch.thenBlock() : branch.elseBlock();
  auto *constantBlock = isAnd ? branch.elseBlock() : branch.thenBlock();
  builder.setInsertionPointToStart(rhsBlock);
  auto rhs = clang_mlir::toBool(builder, loc, generateExpr(expr->getRHS()));
  if (!rhs) {
    return {};
  }
  mlir::scf::YieldOp::create(builder, loc, rhs);
  builder.setInsertionPointToStart(constantBlock);
  mlir::scf::YieldOp::create(builder, loc,
                             clang_mlir::boolConst(builder, loc, !isAnd));
  builder.setInsertionPointAfter(branch);
  return clang_mlir::toValue(builder, loc, branch.getResult(0),
                             convertType(expr->getType()), false);
}

} // namespace cmlir::frontend
