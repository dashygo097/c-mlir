#include "../../../Converter.h"
#include "../../Utils/Casts.h"
#include "../../Utils/LHS.h"
#include "../../Utils/Numerics.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "clang/AST/OperationKinds.h"
#include "llvm/Support/WithColor.h"

namespace cmlirc {

auto emitCompoundArith(mlir::OpBuilder &builder, mlir::Location loc,
                       clang::BinaryOperatorKind op, mlir::Value lhs,
                       mlir::Value rhs) -> mlir::Value {
  using CBO = clang::BinaryOperatorKind;

  switch (op) {
  case CBO::BO_AddAssign:
    return utils::add(builder, loc, lhs, rhs);
  case CBO::BO_SubAssign:
    return utils::sub(builder, loc, lhs, rhs);
  case CBO::BO_MulAssign:
    return utils::mul(builder, loc, lhs, rhs);
  case CBO::BO_DivAssign:
    return utils::divs(builder, loc, lhs, rhs);
  case CBO::BO_RemAssign:
    return utils::rems(builder, loc, lhs, rhs);
  case CBO::BO_AndAssign:
    return utils::bitAnd(builder, loc, lhs, rhs);
  case CBO::BO_OrAssign:
    return utils::bitOr(builder, loc, lhs, rhs);
  case CBO::BO_XorAssign:
    return utils::bitXor(builder, loc, lhs, rhs);
  case CBO::BO_ShlAssign:
    return utils::shl(builder, loc, lhs, rhs);
  case CBO::BO_ShrAssign:
    return utils::shrs(builder, loc, lhs, rhs);
  default:
    llvm::WithColor::error() << "cmlirc: unsupported compound assignment: "
                             << clang::BinaryOperator::getOpcodeStr(op) << "\n";
    return nullptr;
  }
}

auto CMLIRConverter::generateAssignmentBinaryOperator(
    clang::BinaryOperator *assignOp) -> mlir::Value {
  mlir::OpBuilder &builder = contextManager.Builder();
  mlir::Location loc = builder.getUnknownLoc();

  utils::LHSKind lhsKind = utils::classifyLHS(assignOp->getLHS());

  mlir::Value lhsAddr = generateExpr(assignOp->getLHS());
  if (!lhsAddr) {
    return nullptr;
  }

  std::optional<ArrayAccessInfo> arrayAccess;
  if (lhsKind == utils::LHSKind::Indexed) {
    if (!lastArrayAccess) {
      llvm::WithColor::error() << "cmlirc: missing array access info for LHS\n";
      return nullptr;
    }

    arrayAccess = std::move(lastArrayAccess);
    lastArrayAccess.reset();
  }

  mlir::Value rhsValue = generateExpr(assignOp->getRHS());
  if (!rhsValue) {
    return nullptr;
  }

  mlir::Value resultValue = rhsValue;

  if (assignOp->getOpcode() != clang::BO_Assign) {
    mlir::Type lhsElemType = convertType(assignOp->getLHS()->getType());
    if (!lhsElemType) {
      return nullptr;
    }

    if (assignOp->getLHS()->getType()->isPointerType()) {
      lhsElemType = mlir::LLVM::LLVMPointerType::get(builder.getContext());
    }

    mlir::Value oldValue = utils::loadLHS(builder, loc, lhsKind, lhsAddr,
                                          arrayAccess, lhsElemType);
    if (!oldValue) {
      return nullptr;
    }

    mlir::Type lhsType = oldValue.getType();
    mlir::Value computeLHS = oldValue;

    if (auto *compOp =
            mlir::dyn_cast<clang::CompoundAssignOperator>(assignOp)) {
      mlir::Type computeType = convertType(compOp->getComputationResultType());
      if (!computeType) {
        return nullptr;
      }

      bool isSigned =
          compOp->getLHS()->getType()->isSignedIntegerOrEnumerationType();

      computeLHS =
          utils::toValue(builder, loc, computeLHS, computeType, isSigned);
      if (!computeLHS) {
        return nullptr;
      }
    }

    mlir::Value computed = emitCompoundArith(
        builder, loc, assignOp->getOpcode(), computeLHS, rhsValue);
    if (!computed) {
      return nullptr;
    }

    resultValue = computed;

    if (auto *compOp =
            mlir::dyn_cast<clang::CompoundAssignOperator>(assignOp)) {
      bool isSigned =
          compOp->getLHS()->getType()->isSignedIntegerOrEnumerationType();

      resultValue = utils::toValue(builder, loc, computed, lhsType, isSigned);
      if (!resultValue) {
        return nullptr;
      }
    }
  }

  if (assignOp->getOpcode() == clang::BO_Assign) {
    if (assignOp->getLHS()->getType()->isPointerType()) {
      auto ptrType = mlir::LLVM::LLVMPointerType::get(builder.getContext());

      if (!mlir::isa<mlir::LLVM::LLVMPointerType>(resultValue.getType())) {
        resultValue = utils::toPointer(builder, loc, resultValue, ptrType);
        if (!resultValue) {
          return nullptr;
        }
      }
    } else {
      mlir::Type lhsType = convertType(assignOp->getLHS()->getType());
      if (!lhsType) {
        return nullptr;
      }

      if (resultValue.getType() != lhsType) {
        bool isSigned =
            assignOp->getRHS()->getType()->isSignedIntegerOrEnumerationType();

        if (mlir::Value casted =
                utils::toValue(builder, loc, resultValue, lhsType, isSigned)) {
          resultValue = casted;
        }
      }
    }
  }

  utils::storeLHS(builder, loc, lhsKind, resultValue, lhsAddr, arrayAccess);
  return resultValue;
}

} // namespace cmlirc
