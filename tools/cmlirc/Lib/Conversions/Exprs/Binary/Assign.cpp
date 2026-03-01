#include "../../../Converter.h"
#include "../../Utils/Casts.h"
#include "../../Utils/LHS.h"
#include "../../Utils/Operators.h"
#include "clang/AST/OperationKinds.h"

namespace cmlirc {
// Emit the arithmetic for a compound-assignment operator (+=, -= …).
mlir::Value emitCompoundArith(mlir::OpBuilder &builder, mlir::Location loc,
                              clang::BinaryOperatorKind op, mlir::Value lhs,
                              mlir::Value rhs) {
  using CBO = clang::BinaryOperatorKind;

  switch (op) {
  case CBO::BO_AddAssign:
    return detail::emitOp<mlir::arith::AddIOp, mlir::arith::AddFOp>(
        builder, loc, lhs, rhs);
  case CBO::BO_SubAssign:
    return detail::emitOp<mlir::arith::SubIOp, mlir::arith::SubFOp>(
        builder, loc, lhs, rhs);
  case CBO::BO_MulAssign:
    return detail::emitOp<mlir::arith::MulIOp, mlir::arith::MulFOp>(
        builder, loc, lhs, rhs);
  case CBO::BO_DivAssign:
    return detail::emitOp<mlir::arith::DivSIOp, mlir::arith::DivFOp>(
        builder, loc, lhs, rhs);
  case CBO::BO_RemAssign:
    return detail::emitIntOp<mlir::arith::RemSIOp>(builder, loc, lhs, rhs);
  case CBO::BO_AndAssign:
    return detail::emitIntOp<mlir::arith::AndIOp>(builder, loc, lhs, rhs);
  case CBO::BO_OrAssign:
    return detail::emitIntOp<mlir::arith::OrIOp>(builder, loc, lhs, rhs);
  case CBO::BO_XorAssign:
    return detail::emitIntOp<mlir::arith::XOrIOp>(builder, loc, lhs, rhs);
  case CBO::BO_ShlAssign:
    return detail::emitIntOp<mlir::arith::ShLIOp>(builder, loc, lhs, rhs);
  case CBO::BO_ShrAssign:
    return detail::emitIntOp<mlir::arith::ShRSIOp>(builder, loc, lhs, rhs);
  default:
    llvm::errs() << "cmlirc: unsupported compound assignment: "
                 << clang::BinaryOperator::getOpcodeStr(op) << "\n";
    return nullptr;
  }
}

mlir::Value
CMLIRConverter::generateAssignmentBinaryOperator(clang::BinaryOperator *binOp) {
  mlir::OpBuilder &builder = context_manager_.Builder();
  mlir::Location loc = builder.getUnknownLoc();

  detail::LHSKind lhsKind = detail::classifyLHS(binOp->getLHS());

  // Evaluate LHS address
  mlir::Value lhsAddr = generateExpr(binOp->getLHS());
  if (!lhsAddr) {
    llvm::errs() << "cmlirc: failed to generate LHS address\n";
    return nullptr;
  }

  // For indexed access, consume the side-channel ArrayAccessInfo immediately.
  std::optional<ArrayAccessInfo> arrayAccess;
  if (lhsKind == detail::LHSKind::Indexed) {
    if (!lastArrayAccess) {
      llvm::errs() << "cmlirc: missing array access info for LHS\n";
      return nullptr;
    }
    arrayAccess = std::move(lastArrayAccess);
    lastArrayAccess.reset();
  }

  // Evaluate RHS value
  mlir::Value rhsValue = generateExpr(binOp->getRHS());
  if (!rhsValue) {
    llvm::errs() << "cmlirc: failed to generate RHS\n";
    return nullptr;
  }

  mlir::Value resultValue = rhsValue;

  // Compound-assignment: read-modify-write
  if (binOp->getOpcode() != clang::BO_Assign) {
    mlir::Value oldValue = loadLHS(builder, loc, lhsKind, lhsAddr, arrayAccess);
    mlir::Type lhsType = oldValue.getType();
    mlir::Value computeLHS = oldValue;

    // Promote LHS to the computation type if required (e.g. short += int).
    if (auto *compOp = mlir::dyn_cast<clang::CompoundAssignOperator>(binOp)) {
      mlir::Type computeType = convertType(compOp->getComputationResultType());
      bool isSigned = compOp->getLHS()->getType()->isSignedIntegerType();
      computeLHS =
          detail::promoteValue(builder, loc, computeLHS, computeType, isSigned);
    }

    mlir::Value computed = emitCompoundArith(builder, loc, binOp->getOpcode(),
                                             computeLHS, rhsValue);
    if (!computed)
      return nullptr;

    // Truncate back to the LHS storage type if computation widened it.
    resultValue = detail::truncateValue(builder, loc, computed, lhsType);
  }

  // Store result
  storeLHS(builder, loc, lhsKind, resultValue, lhsAddr, arrayAccess);
  return resultValue;
}

} // namespace cmlirc
