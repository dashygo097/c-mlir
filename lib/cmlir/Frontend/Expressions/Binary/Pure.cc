#include "../../Lowering.h"
#include "clang-mlir/Casts.h"
#include "clang-mlir/Constants.h"
#include "clang-mlir/Numerics.h"
#include "clang-mlir/Operators.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "clang/AST/OperationKinds.h"
#include "llvm/Support/WithColor.h"

namespace cmlir::frontend {

// Handle pure value-producing binary expressions (no side effects on LHS).
auto ASTLowering::generatePureBinaryOperator(clang::BinaryOperator *binOp)
    -> mlir::Value {
  mlir::OpBuilder &builder = context.builder();
  mlir::Location loc = context.source().location(binOp->getExprLoc());

  if (binOp->getOpcode() == clang::BO_LAnd ||
      binOp->getOpcode() == clang::BO_LOr) {
    return generateLogicalOperator(binOp);
  }
  mlir::Value lhs = generateExpr(binOp->getLHS());
  mlir::Value rhs = generateExpr(binOp->getRHS());
  if (!lhs || !rhs) {
    return {};
  }

  using CBO = clang::BinaryOperatorKind;
  using IP = mlir::arith::CmpIPredicate;
  using FP = mlir::arith::CmpFPredicate;

  switch (binOp->getOpcode()) {
  // Arithmetic
  case CBO::BO_Add:
    return clang_mlir::add(builder, loc, lhs, rhs);
  case CBO::BO_Sub:
    return clang_mlir::sub(builder, loc, lhs, rhs);
  case CBO::BO_Mul:
    return clang_mlir::mul(builder, loc, lhs, rhs);
  case CBO::BO_Div:
    return clang_mlir::divs(builder, loc, lhs, rhs);
  case CBO::BO_Rem:
    return clang_mlir::rems(builder, loc, lhs, rhs);

  // Bitwise
  case CBO::BO_And:
    return clang_mlir::bitAnd(builder, loc, lhs, rhs);
  case CBO::BO_Or:
    return clang_mlir::bitOr(builder, loc, lhs, rhs);
  case CBO::BO_Xor:
    return clang_mlir::bitXor(builder, loc, lhs, rhs);
  case CBO::BO_Shl:
    return clang_mlir::shl(builder, loc, lhs, rhs);
  case CBO::BO_Shr:
    return clang_mlir::shrs(builder, loc, lhs, rhs);

  // Cmp
  case CBO::BO_LT:
    return clang_mlir::toValue(
        builder, loc,
        clang_mlir::emitCmpOp(builder, loc, IP::slt, FP::OLT, lhs, rhs),
        convertType(binOp->getType()), false);
  case CBO::BO_LE:
    return clang_mlir::toValue(
        builder, loc,
        clang_mlir::emitCmpOp(builder, loc, IP::sle, FP::OLE, lhs, rhs),
        convertType(binOp->getType()), false);
  case CBO::BO_GT:
    return clang_mlir::toValue(
        builder, loc,
        clang_mlir::emitCmpOp(builder, loc, IP::sgt, FP::OGT, lhs, rhs),
        convertType(binOp->getType()), false);
  case CBO::BO_GE:
    return clang_mlir::toValue(
        builder, loc,
        clang_mlir::emitCmpOp(builder, loc, IP::sge, FP::OGE, lhs, rhs),
        convertType(binOp->getType()), false);
  case CBO::BO_EQ:
    return clang_mlir::toValue(
        builder, loc,
        clang_mlir::emitCmpOp(builder, loc, IP::eq, FP::OEQ, lhs, rhs),
        convertType(binOp->getType()), false);
  case CBO::BO_NE:
    return clang_mlir::toValue(
        builder, loc,
        clang_mlir::emitCmpOp(builder, loc, IP::ne, FP::UNE, lhs, rhs),
        convertType(binOp->getType()), false);

  default:
    llvm::WithColor::error()
        << "clang-to-mlir: unsupported binary operator: "
        << clang::BinaryOperator::getOpcodeStr(binOp->getOpcode()) << "\n";
    return nullptr;
  }
}

} // namespace cmlir::frontend
