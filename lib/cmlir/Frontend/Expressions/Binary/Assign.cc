#include "../../Lowering.h"
#include "clang-mlir/Casts.h"
#include "clang-mlir/Numerics.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "clang/AST/OperationKinds.h"
#include "llvm/Support/WithColor.h"

namespace cmlir::frontend {

auto emitCompoundArith(mlir::OpBuilder &builder, mlir::Location loc,
                       clang::BinaryOperatorKind op, mlir::Value lhs,
                       mlir::Value rhs, bool isUnsigned) -> mlir::Value {
  using CBO = clang::BinaryOperatorKind;

  switch (op) {
  case CBO::BO_AddAssign:
    return clang_mlir::add(builder, loc, lhs, rhs);
  case CBO::BO_SubAssign:
    return clang_mlir::sub(builder, loc, lhs, rhs);
  case CBO::BO_MulAssign:
    return clang_mlir::mul(builder, loc, lhs, rhs);
  case CBO::BO_DivAssign:
    return isUnsigned ? clang_mlir::divu(builder, loc, lhs, rhs)
                      : clang_mlir::divs(builder, loc, lhs, rhs);
  case CBO::BO_RemAssign:
    return isUnsigned ? clang_mlir::remu(builder, loc, lhs, rhs)
                      : clang_mlir::rems(builder, loc, lhs, rhs);
  case CBO::BO_AndAssign:
    return clang_mlir::bitAnd(builder, loc, lhs, rhs);
  case CBO::BO_OrAssign:
    return clang_mlir::bitOr(builder, loc, lhs, rhs);
  case CBO::BO_XorAssign:
    return clang_mlir::bitXor(builder, loc, lhs, rhs);
  case CBO::BO_ShlAssign:
    return clang_mlir::shl(builder, loc, lhs, rhs);
  case CBO::BO_ShrAssign:
    return isUnsigned ? clang_mlir::shru(builder, loc, lhs, rhs)
                      : clang_mlir::shrs(builder, loc, lhs, rhs);
  default:
    llvm::WithColor::error()
        << "clang-to-mlir: unsupported compound assignment: "
        << clang::BinaryOperator::getOpcodeStr(op) << "\n";
    return nullptr;
  }
}

auto ASTLowering::emitAssignmentAddress(clang::BinaryOperator *assignOp)
    -> mlir::FailureOr<clang_mlir::Address> {
  mlir::OpBuilder &builder = context.builder();
  mlir::Location loc = context.source().location(assignOp->getExprLoc());

  mlir::Value rhsValue;
  // C++17 sequences the RHS before the LHS, including compound assignment.
  if (context.ast().getLangOpts().CPlusPlus17) {
    rhsValue = generateExpr(assignOp->getRHS());
    if (!rhsValue) {
      return mlir::failure();
    }
  }
  auto address = emitAddress(assignOp->getLHS());
  if (mlir::failed(address)) {
    return mlir::failure();
  }

  if (!rhsValue) {
    rhsValue = generateExpr(assignOp->getRHS());
  }
  if (!rhsValue) {
    return mlir::failure();
  }

  mlir::Value resultValue = rhsValue;

  if (assignOp->getOpcode() != clang::BO_Assign) {
    mlir::Type lhsElemType = types.convertType(assignOp->getLHS()->getType());
    if (!lhsElemType) {
      return mlir::failure();
    }

    if (assignOp->getLHS()->getType()->isPointerType()) {
      lhsElemType = mlir::LLVM::LLVMPointerType::get(builder.getContext());
    }

    mlir::Value oldValue = address->load(builder, loc);
    if (!oldValue) {
      return mlir::failure();
    }

    mlir::Type lhsType = oldValue.getType();
    mlir::Value computeLHS = oldValue;
    clang::QualType computationType = assignOp->getLHS()->getType();

    if (auto *compOp =
            mlir::dyn_cast<clang::CompoundAssignOperator>(assignOp)) {
      // Select operations using the promoted computation type, before the
      // result is converted back to the storage type of the LHS.
      computationType = compOp->getComputationLHSType();
      mlir::Type computeType = types.convertType(computationType);
      if (!computeType) {
        return mlir::failure();
      }

      bool isSigned =
          compOp->getLHS()->getType()->isSignedIntegerOrEnumerationType();

      computeLHS =
          clang_mlir::toValue(builder, loc, computeLHS, computeType, isSigned);
      if (!computeLHS) {
        return mlir::failure();
      }
    }

    if (assignOp->getOpcode() == clang::BO_ShlAssign ||
        assignOp->getOpcode() == clang::BO_ShrAssign) {
      rhsValue = clang_mlir::toValue(
          builder, loc, rhsValue, computeLHS.getType(),
          assignOp->getRHS()->getType()->isSignedIntegerOrEnumerationType());
      if (!rhsValue) {
        return mlir::failure();
      }
    }

    mlir::Value computed = emitCompoundArith(
        builder, loc, assignOp->getOpcode(), computeLHS, rhsValue,
        computationType->isUnsignedIntegerOrEnumerationType());
    if (!computed) {
      return mlir::failure();
    }

    resultValue = computed;

    if (auto *compOp =
            mlir::dyn_cast<clang::CompoundAssignOperator>(assignOp)) {
      bool isSigned =
          compOp->getLHS()->getType()->isSignedIntegerOrEnumerationType();

      resultValue =
          clang_mlir::toValue(builder, loc, computed, lhsType, isSigned);
      if (!resultValue) {
        return mlir::failure();
      }
    }
  }

  if (assignOp->getOpcode() == clang::BO_Assign) {
    if (assignOp->getLHS()->getType()->isPointerType()) {
      auto ptrType = mlir::LLVM::LLVMPointerType::get(builder.getContext());

      if (!mlir::isa<mlir::LLVM::LLVMPointerType>(resultValue.getType())) {
        resultValue = clang_mlir::toPointer(builder, loc, resultValue, ptrType);
        if (!resultValue) {
          return mlir::failure();
        }
      }
    } else {
      mlir::Type lhsType = types.convertType(assignOp->getLHS()->getType());
      if (!lhsType) {
        return mlir::failure();
      }

      if (resultValue.getType() != lhsType) {
        bool isSigned =
            assignOp->getRHS()->getType()->isSignedIntegerOrEnumerationType();

        if (mlir::Value casted = clang_mlir::toValue(builder, loc, resultValue,
                                                     lhsType, isSigned)) {
          resultValue = casted;
        }
      }
    }
  }

  if (mlir::failed(address->store(builder, loc, resultValue))) {
    return mlir::failure();
  }
  return *address;
}

auto ASTLowering::generateAssignmentBinaryOperator(clang::BinaryOperator *expr)
    -> mlir::Value {
  auto address = emitAssignmentAddress(expr);
  if (mlir::failed(address)) {
    return {};
  }
  return address->load(context.builder(),
                       context.source().location(expr->getOperatorLoc()));
}

} // namespace cmlir::frontend
