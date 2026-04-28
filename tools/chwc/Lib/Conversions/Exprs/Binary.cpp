#include "../../Converter.h"
#include "../Utils/HWOps.h"
#include "llvm/Support/WithColor.h"

namespace chwc {

auto CHWConverter::generateBinaryOperator(clang::BinaryOperator *binOp)
    -> mlir::Value {
  if (binOp->isAssignmentOp()) {
    return generateAssignmentBinaryOperator(binOp);
  }

  mlir::OpBuilder &builder = contextManager.Builder();
  mlir::Location loc = builder.getUnknownLoc();

  mlir::Value lhs = generateExpr(binOp->getLHS());
  mlir::Value rhs = generateExpr(binOp->getRHS());

  if (!lhs || !rhs) {
    return nullptr;
  }

  switch (binOp->getOpcode()) {
  case clang::BO_Add:
    return utils::add(builder, loc, lhs, rhs);

  case clang::BO_Sub:
    return utils::sub(builder, loc, lhs, rhs);

  case clang::BO_Mul:
    return utils::mul(builder, loc, lhs, rhs);

  case clang::BO_And:
    return utils::bitAnd(builder, loc, lhs, rhs);

  case clang::BO_Or:
    return utils::bitOr(builder, loc, lhs, rhs);

  case clang::BO_Xor:
    return utils::bitXor(builder, loc, lhs, rhs);

  case clang::BO_EQ:
    return utils::icmpEq(builder, loc, lhs, rhs);

  case clang::BO_NE:
    return utils::icmpNe(builder, loc, lhs, rhs);

  case clang::BO_LT:
    return utils::icmpSlt(builder, loc, lhs, rhs);

  case clang::BO_LE:
    return utils::icmpSle(builder, loc, lhs, rhs);

  case clang::BO_GT:
    return utils::icmpSgt(builder, loc, lhs, rhs);

  case clang::BO_GE:
    return utils::icmpSge(builder, loc, lhs, rhs);

  default:
    llvm::WithColor::error() << "chwc: unsupported binary operator\n";
    return nullptr;
  }
}

} // namespace chwc
