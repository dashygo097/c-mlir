#include "../../../Converter.h"
#include "../../Utils/Cast.h"
#include "../../Utils/Comb.h"
#include "clang/AST/OperationKinds.h"
#include "llvm/Support/WithColor.h"

namespace chwc {

auto CHWConverter::generatePureBinaryOperator(clang::BinaryOperator *binOp)
    -> mlir::Value {
  mlir::OpBuilder &builder = contextManager.Builder();
  mlir::Location loc = builder.getUnknownLoc();

  mlir::Value lhs = generateExpr(binOp->getLHS());
  mlir::Value rhs = generateExpr(binOp->getRHS());

  if (!lhs || !rhs) {
    llvm::WithColor::error() << "chwc: failed to generate binary operands\n";
    return nullptr;
  }

  using CBO = clang::BinaryOperatorKind;

  switch (binOp->getOpcode()) {
  case CBO::BO_Add:
    return utils::add(builder, loc, lhs, rhs);

  case CBO::BO_Sub:
    return utils::sub(builder, loc, lhs, rhs);

  case CBO::BO_Mul:
    return utils::mul(builder, loc, lhs, rhs);

  case CBO::BO_Div:
    return utils::div(builder, loc, lhs, rhs);

  case CBO::BO_Rem:
    return utils::mod(builder, loc, lhs, rhs);

  case CBO::BO_And:
    return utils::bitAnd(builder, loc, lhs, rhs);

  case CBO::BO_Or:
    return utils::bitOr(builder, loc, lhs, rhs);

  case CBO::BO_Xor:
    return utils::bitXor(builder, loc, lhs, rhs);

  case CBO::BO_Shl:
    return utils::shl(builder, loc, lhs, rhs);

  case CBO::BO_Shr:
    return utils::shrU(builder, loc, lhs, rhs);

  case CBO::BO_EQ:
    return utils::icmpEq(builder, loc, lhs, rhs);

  case CBO::BO_NE:
    return utils::icmpNe(builder, loc, lhs, rhs);

  case CBO::BO_LT:
    return utils::icmpUlt(builder, loc, lhs, rhs);

  case CBO::BO_LE:
    return utils::icmpUle(builder, loc, lhs, rhs);

  case CBO::BO_GT:
    return utils::icmpUgt(builder, loc, lhs, rhs);

  case CBO::BO_GE:
    return utils::icmpUge(builder, loc, lhs, rhs);

  case CBO::BO_LAnd:
    return generateLAndBinaryOperator(lhs, rhs);

  case CBO::BO_LOr:
    return generateLOrBinaryOperator(lhs, rhs);

  default:
    llvm::WithColor::error()
        << "chwc: unsupported binary operator: "
        << clang::BinaryOperator::getOpcodeStr(binOp->getOpcode()) << "\n";
    return nullptr;
  }
}

auto CHWConverter::generateLAndBinaryOperator(mlir::Value lhs, mlir::Value rhs)
    -> mlir::Value {
  mlir::OpBuilder &builder = contextManager.Builder();
  mlir::Location loc = builder.getUnknownLoc();

  lhs = utils::toBool(builder, loc, lhs);
  rhs = utils::toBool(builder, loc, rhs);

  if (!lhs || !rhs) {
    return nullptr;
  }

  return utils::bitAnd(builder, loc, lhs, rhs);
}

auto CHWConverter::generateLOrBinaryOperator(mlir::Value lhs, mlir::Value rhs)
    -> mlir::Value {
  mlir::OpBuilder &builder = contextManager.Builder();
  mlir::Location loc = builder.getUnknownLoc();

  lhs = utils::toBool(builder, loc, lhs);
  rhs = utils::toBool(builder, loc, rhs);

  if (!lhs || !rhs) {
    return nullptr;
  }

  return utils::bitOr(builder, loc, lhs, rhs);
}

} // namespace chwc
