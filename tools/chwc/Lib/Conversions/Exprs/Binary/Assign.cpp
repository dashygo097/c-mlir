#include "../../../Converter.h"
#include "../../Utils/Cast.h"
#include "../../Utils/Comb.h"
#include "../../Utils/Expr.h"
#include "clang/AST/OperationKinds.h"
#include "llvm/Support/WithColor.h"

namespace chwc {
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
    return utils::div(builder, loc, lhs, rhs);

  case CBO::BO_RemAssign:
    return utils::mod(builder, loc, lhs, rhs);

  case CBO::BO_AndAssign:
    return utils::bitAnd(builder, loc, lhs, rhs);

  case CBO::BO_OrAssign:
    return utils::bitOr(builder, loc, lhs, rhs);

  case CBO::BO_XorAssign:
    return utils::bitXor(builder, loc, lhs, rhs);

  case CBO::BO_ShlAssign:
    return utils::shl(builder, loc, lhs, rhs);

  case CBO::BO_ShrAssign:
    return utils::shr(builder, loc, lhs, rhs);

  default:
    llvm::WithColor::error() << "chwc: unsupported compound assignment: "
                             << clang::BinaryOperator::getOpcodeStr(op) << "\n";
    return nullptr;
  }
}

auto CHWConverter::generateAssignmentBinaryOperator(
    clang::BinaryOperator *assignOp) -> mlir::Value {
  mlir::OpBuilder &builder = contextManager.Builder();
  mlir::Location loc = builder.getUnknownLoc();

  mlir::Value rhsValue = generateExpr(assignOp->getRHS());
  if (!rhsValue) {
    llvm::WithColor::error() << "chwc: failed to generate RHS\n";
    return nullptr;
  }

  mlir::Value resultValue = rhsValue;

  if (assignOp->getOpcode() != clang::BO_Assign) {
    auto *compoundOp = mlir::dyn_cast<clang::CompoundAssignOperator>(assignOp);
    if (!compoundOp) {
      llvm::WithColor::error() << "chwc: expected CompoundAssignOperator\n";
      return nullptr;
    }

    mlir::Value oldValue = generateExpr(assignOp->getLHS());
    if (!oldValue) {
      llvm::WithColor::error()
          << "chwc: failed to generate compound assignment lhs\n";
      return nullptr;
    }

    mlir::Type lhsStorageType = oldValue.getType();
    mlir::Type computeType =
        convertType(compoundOp->getComputationResultType());

    mlir::Value computeLHS =
        utils::promoteValue(builder, loc, oldValue, computeType);
    mlir::Value computeRHS =
        utils::promoteValue(builder, loc, rhsValue, computeType);

    mlir::Value computed = emitCompoundArith(
        builder, loc, assignOp->getOpcode(), computeLHS, computeRHS);

    if (!computed) {
      return nullptr;
    }

    resultValue = utils::truncateValue(builder, loc, computed, lhsStorageType);
  }

  const clang::FieldDecl *fieldDecl = getAssignedField(assignOp->getLHS());

  if (fieldDecl) {
    auto fieldIt = fieldTable.find(fieldDecl);
    if (fieldIt == fieldTable.end()) {
      llvm::WithColor::error()
          << "chwc: assignment lhs is not hardware field\n";
      return resultValue;
    }

    HWFieldInfo &fieldInfo = fieldIt->second;

    switch (fieldInfo.kind) {
    case HWFieldKind::Input:
      llvm::WithColor::error() << "chwc: cannot assign to hardware input\n";
      break;

    case HWFieldKind::Output:
      outputValueTable[fieldDecl] = resultValue;
      break;

    case HWFieldKind::Reg:
      nextFieldValueTable[fieldDecl] = resultValue;
      break;

    case HWFieldKind::Wire:
      currentFieldValueTable[fieldDecl] = resultValue;
      break;
    }

    return resultValue;
  }

  auto *declRef = mlir::dyn_cast_or_null<clang::DeclRefExpr>(
      utils::ignoreCasts(assignOp->getLHS()));

  if (declRef) {
    if (auto *varDecl = mlir::dyn_cast<clang::VarDecl>(declRef->getDecl())) {
      localValueTable[varDecl] = resultValue;
      return resultValue;
    }
  }

  llvm::WithColor::error() << "chwc: unsupported assignment lhs\n";
  return resultValue;
}

auto CHWConverter::generateCompoundAssignmentBinaryOperator(
    clang::CompoundAssignOperator *compoundOp) -> mlir::Value {
  return generateAssignmentBinaryOperator(compoundOp);
}

} // namespace chwc
