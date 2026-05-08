#include "../../Converter.h"
#include "../Utils/Cast.h"
#include "../Utils/Comb.h"
#include "../Utils/Constant.h"
#include "../Utils/Type.h"
#include "llvm/Support/WithColor.h"

namespace chwc {

auto CHWConverter::generatePureBinaryOperator(clang::BinaryOperator *binOp)
    -> mlir::Value {
  if (!binOp) {
    return nullptr;
  }

  mlir::OpBuilder &builder = contextManager.Builder();
  mlir::Location loc = builder.getUnknownLoc();

  mlir::Value lhs = generateExpr(binOp->getLHS());
  if (!lhs) {
    llvm::WithColor::error() << "chwc: failed to generate binary lhs\n";
    return nullptr;
  }

  mlir::Value rhs = nullptr;

  clang::Expr *rhsExpr = binOp->getRHS()->IgnoreParenImpCasts();
  if (auto *rhsLit = mlir::dyn_cast<clang::IntegerLiteral>(rhsExpr)) {
    rhs = utils::intConst(builder, loc, lhs.getType(),
                          rhsLit->getValue().getSExtValue());
  } else {
    rhs = generateExpr(binOp->getRHS());
  }

  if (!rhs) {
    llvm::WithColor::error() << "chwc: failed to generate binary rhs\n";
    return nullptr;
  }

  if (lhs.getType() != rhs.getType()) {
    rhs = utils::promoteValue(builder, loc, rhs, lhs.getType());
    if (!rhs) {
      return nullptr;
    }
  }

  bool isSigned = utils::isSignedType(binOp->getLHS()->getType()) ||
                  utils::isSignedType(binOp->getRHS()->getType());

  bool isSignedShift = utils::isSignedType(binOp->getLHS()->getType());

  switch (binOp->getOpcode()) {
  case clang::BO_Add:
    return utils::add(builder, loc, lhs, rhs);

  case clang::BO_Sub:
    return utils::sub(builder, loc, lhs, rhs);

  case clang::BO_Mul:
    return utils::mul(builder, loc, lhs, rhs);

  case clang::BO_Div:
    return isSigned ? utils::divS(builder, loc, lhs, rhs)
                    : utils::divU(builder, loc, lhs, rhs);

  case clang::BO_Rem:
    return isSigned ? utils::modS(builder, loc, lhs, rhs)
                    : utils::modU(builder, loc, lhs, rhs);

  case clang::BO_And:
    return utils::bitAnd(builder, loc, lhs, rhs);

  case clang::BO_Or:
    return utils::bitOr(builder, loc, lhs, rhs);

  case clang::BO_Xor:
    return utils::bitXor(builder, loc, lhs, rhs);

  case clang::BO_Shl:
    return utils::shl(builder, loc, lhs, rhs);

  case clang::BO_Shr:
    return isSignedShift ? utils::shrS(builder, loc, lhs, rhs)
                         : utils::shrU(builder, loc, lhs, rhs);

  case clang::BO_EQ:
    return utils::icmpEq(builder, loc, lhs, rhs);

  case clang::BO_NE:
    return utils::icmpNe(builder, loc, lhs, rhs);

  case clang::BO_LT:
    return isSigned ? utils::icmpSlt(builder, loc, lhs, rhs)
                    : utils::icmpUlt(builder, loc, lhs, rhs);

  case clang::BO_LE:
    return isSigned ? utils::icmpSle(builder, loc, lhs, rhs)
                    : utils::icmpUle(builder, loc, lhs, rhs);

  case clang::BO_GT:
    return isSigned ? utils::icmpSgt(builder, loc, lhs, rhs)
                    : utils::icmpUgt(builder, loc, lhs, rhs);

  case clang::BO_GE:
    return isSigned ? utils::icmpSge(builder, loc, lhs, rhs)
                    : utils::icmpUge(builder, loc, lhs, rhs);

  case clang::BO_LAnd:
    return generateLAndBinaryOperator(lhs, rhs);

  case clang::BO_LOr:
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
