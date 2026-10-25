#include "../ExpressionLowering.h"
#include "../PointerArithmetic.h"
#include "clang-mlir/Casts.h"
#include "clang-mlir/FloatingPoint.h"
#include "clang-mlir/Numerics.h"
#include "clang/AST/OperationKinds.h"
#include "llvm/Support/WithColor.h"

namespace cmlir::frontend {

auto emitCompoundArith(mlir::OpBuilder &builder, mlir::Location loc,
                       clang::BinaryOperatorKind op, mlir::Value lhs,
                       mlir::Value rhs, bool isUnsigned,
                       mlir::arith::FastMathFlags flags) -> mlir::Value {
  using CBO = clang::BinaryOperatorKind;

  switch (op) {
  case CBO::BO_AddAssign:
    return clang_mlir::add(builder, loc, lhs, rhs, flags);
  case CBO::BO_SubAssign:
    return clang_mlir::sub(builder, loc, lhs, rhs, flags);
  case CBO::BO_MulAssign:
    return clang_mlir::mul(builder, loc, lhs, rhs, flags);
  case CBO::BO_DivAssign:
    return isUnsigned ? clang_mlir::divu(builder, loc, lhs, rhs, flags)
                      : clang_mlir::divs(builder, loc, lhs, rhs, flags);
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

auto ExpressionLowering::emitAssignmentLValue(clang::BinaryOperator &expr,
                                              mlir::Type resultType,
                                              LValueConsumer consume)
    -> mlir::FailureOr<mlir::Value> {
  // C++17 sequences the RHS before all LHS evaluation, including its condition.
  mlir::Value rhs;
  if (context.ast().getLangOpts().CPlusPlus17) {
    rhs = generateExpr(expr.getRHS());
    if (!rhs) {
      return mlir::failure();
    }
  }
  return emitLValue(
      expr.getLHS(), resultType,
      [&](const LValueResult &result) -> mlir::FailureOr<mlir::Value> {
        const auto *object = std::get_if<ObjectResult>(&result);
        if (!object) {
          return mlir::failure();
        }
        auto value = storeAssignment(&expr, object->address,
                                     rhs ? rhs : generateExpr(expr.getRHS()));
        if (!value) {
          return mlir::failure();
        }
        return consume(ObjectResult{object->address, value});
      });
}

auto ExpressionLowering::storeAssignment(clang::BinaryOperator *assignOp,
                                         const clang_mlir::Address &address,
                                         mlir::Value rhsValue) -> mlir::Value {
  auto &builder = context.builder();
  auto loc = context.source().location(assignOp->getExprLoc());
  if (!rhsValue) {
    return {};
  }
  mlir::Value resultValue = rhsValue;

  if (assignOp->getOpcode() != clang::BO_Assign) {
    mlir::Value oldValue = address.load(builder, loc);
    if (!oldValue) {
      return {};
    }

    if (assignOp->getLHS()->getType()->isPointerType()) {
      auto element = types.convertStorageType(
          assignOp->getLHS()->getType()->getPointeeType());
      if (!element) {
        return {};
      }
      resultValue = emitPointerOffset(
          builder, loc, oldValue, rhsValue, element,
          assignOp->getRHS()->getType()->isSignedIntegerOrEnumerationType(),
          assignOp->getOpcode() == clang::BO_SubAssign);
    } else {
      mlir::Value computeLHS = oldValue;
      clang::QualType computationType = assignOp->getLHS()->getType();

      if (auto *compOp =
              mlir::dyn_cast<clang::CompoundAssignOperator>(assignOp)) {
        computationType = compOp->getComputationLHSType();
        mlir::Type computeType = types.convertType(computationType);
        if (!computeType) {
          return {};
        }

        bool isSigned =
            compOp->getLHS()->getType()->isSignedIntegerOrEnumerationType();

        computeLHS = clang_mlir::toValue(builder, loc, computeLHS, computeType,
                                         isSigned);
        if (!computeLHS) {
          return {};
        }
      }

      if (assignOp->getOpcode() == clang::BO_ShlAssign ||
          assignOp->getOpcode() == clang::BO_ShrAssign) {
        rhsValue = clang_mlir::toValue(
            builder, loc, rhsValue, computeLHS.getType(),
            assignOp->getRHS()->getType()->isSignedIntegerOrEnumerationType());
        if (!rhsValue) {
          return {};
        }
      }

      mlir::Value computed = emitCompoundArith(
          builder, loc, assignOp->getOpcode(), computeLHS, rhsValue,
          computationType->isUnsignedIntegerOrEnumerationType(),
          clang_mlir::getFastMathFlags(*assignOp, context.ast().getLangOpts()));
      if (!computed) {
        return {};
      }

      resultValue = computed;
    }
  }

  auto sourceType = assignOp->getOpcode() == clang::BO_Assign
                        ? assignOp->getRHS()->getType()
                        : assignOp->getLHS()->getType();
  resultValue =
      clang_mlir::toValue(builder, loc, resultValue, address.elementType(),
                          sourceType->isSignedIntegerOrEnumerationType());
  if (!resultValue) {
    return {};
  }

  if (mlir::failed(address.store(builder, loc, resultValue))) {
    return {};
  }
  return resultValue;
}

auto ExpressionLowering::generateAssignmentBinaryOperator(
    clang::BinaryOperator *expr) -> mlir::Value {
  return loadLValue(expr);
}

} // namespace cmlir::frontend
