#include "../../../Converter.h"
#include "../../Utils/Casts.h"
#include "../../Utils/LHS.h"
#include "../../Utils/Numerics.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "clang/AST/OperationKinds.h"
#include "llvm/Support/WithColor.h"

namespace cmlirc {
// Emit the arithmetic for a compound-assignment operator (+=, -= …).
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

  // Evaluate LHS address
  mlir::Value lhsAddr = generateExpr(assignOp->getLHS());

  // For indexed access, consume the side-channel ArrayAccessInfo immediately.
  std::optional<ArrayAccessInfo> arrayAccess;
  if (lhsKind == utils::LHSKind::Indexed) {
    if (!lastArrayAccess) {
      llvm::WithColor::error() << "cmlirc: missing array access info for LHS\n";
      return nullptr;
    }
    arrayAccess = std::move(lastArrayAccess);
    lastArrayAccess.reset();
  }

  // Evaluate RHS value
  mlir::Value rhsValue = generateExpr(assignOp->getRHS());
  mlir::Value resultValue = rhsValue;

  // Compound-assignment
  if (assignOp->getOpcode() != clang::BO_Assign) {
    mlir::Type lhsElemType = convertType(assignOp->getLHS()->getType());
    mlir::Value oldValue =
        loadLHS(builder, loc, lhsKind, lhsAddr, arrayAccess, lhsElemType);
    mlir::Type lhsType = oldValue.getType();
    mlir::Value computeLHS = oldValue;

    // Promote LHS to the computation type if required (e.g. short += int).
    if (auto *compOp =
            mlir::dyn_cast<clang::CompoundAssignOperator>(assignOp)) {
      mlir::Type computeType = convertType(compOp->getComputationResultType());
      bool isSigned = compOp->getLHS()->getType()->isSignedIntegerType();
      computeLHS =
          utils::toValue(builder, loc, computeLHS, computeType, isSigned);
    }

    mlir::Value computed = emitCompoundArith(
        builder, loc, assignOp->getOpcode(), computeLHS, rhsValue);
    if (!computed) {
      return nullptr;
    }

    // Truncate back to the LHS storage type if computation widened it.
    if (auto *compOp =
            mlir::dyn_cast<clang::CompoundAssignOperator>(assignOp)) {
      bool isSigned = compOp->getLHS()->getType()->isSignedIntegerType();
      resultValue = utils::toValue(builder, loc, computed, lhsType, isSigned);
    }
  }

  // Store result
  storeLHS(builder, loc, lhsKind, resultValue, lhsAddr, arrayAccess);
  return resultValue;
}

} // namespace cmlirc
