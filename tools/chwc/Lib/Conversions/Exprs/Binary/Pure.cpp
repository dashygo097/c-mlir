#include "../../../Converter.h"
#include "../../Utils/Cast.h"
#include "../../Utils/Comb.h"
#include "../../Utils/Type.h"
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

  mlir::Type computeType = lhs.getType();
  if (computeType != rhs.getType()) {
    rhs = utils::promoteValue(builder, loc, rhs, computeType);
  }

  clang::Expr *objectExpr = binOp->getLHS();
  clang::QualType objectType =
      objectExpr ? objectExpr->getType() : clang::QualType{};
  auto typeInfo = utils::getSignalTypeInfo(objectType);

  using CBO = clang::BinaryOperatorKind;

  switch (binOp->getOpcode()) {
  case CBO::BO_Add:
    return utils::add(builder, loc, lhs, rhs);

  case CBO::BO_Sub:
    return utils::sub(builder, loc, lhs, rhs);

  case CBO::BO_Mul:
    return utils::mul(builder, loc, lhs, rhs);

  case CBO::BO_Div:
    return typeInfo.isSigned ? utils::divS(builder, loc, lhs, rhs)
                             : utils::divU(builder, loc, lhs, rhs);

  case CBO::BO_Rem:
    return typeInfo.isSigned ? utils::modS(builder, loc, lhs, rhs)
                             : utils::modU(builder, loc, lhs, rhs);

  case CBO::BO_And:
    return utils::bitAnd(builder, loc, lhs, rhs);

  case CBO::BO_Or:
    return utils::bitOr(builder, loc, lhs, rhs);

  case CBO::BO_Xor:
    return utils::bitXor(builder, loc, lhs, rhs);

  case CBO::BO_Shl:
    return utils::shl(builder, loc, lhs, rhs);

  case CBO::BO_Shr:
    return typeInfo.isSigned ? utils::shrS(builder, loc, lhs, rhs)
                             : utils::shrU(builder, loc, lhs, rhs);

  case CBO::BO_EQ:
    return utils::icmpEq(builder, loc, lhs, rhs);

  case CBO::BO_NE:
    return utils::icmpNe(builder, loc, lhs, rhs);

  case CBO::BO_LT:
    return typeInfo.isSigned ? utils::icmpSlt(builder, loc, lhs, rhs)
                             : utils::icmpUlt(builder, loc, lhs, rhs);

  case CBO::BO_LE:
    return typeInfo.isSigned ? utils::icmpSle(builder, loc, lhs, rhs)
                             : utils::icmpUle(builder, loc, lhs, rhs);

  case CBO::BO_GT:
    return typeInfo.isSigned ? utils::icmpSgt(builder, loc, lhs, rhs)
                             : utils::icmpUgt(builder, loc, lhs, rhs);

  case CBO::BO_GE:
    return typeInfo.isSigned ? utils::icmpSge(builder, loc, lhs, rhs)
                             : utils::icmpUge(builder, loc, lhs, rhs);

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
