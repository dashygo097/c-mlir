#include "ExpressionLowering.h"
#include "clang-mlir/Casts.h"

namespace cmlir::frontend {

auto ExpressionLowering::generateCastExpr(clang::CastExpr *expr)
    -> mlir::Value {
  auto &builder = context.builder();
  auto loc = context.source().location(expr->getExprLoc());
  auto *operand = expr->getSubExpr();
  using CK = clang::CastKind;
  switch (expr->getCastKind()) {
  case CK::CK_LValueToRValue:
    return loadLValue(operand);
  case CK::CK_NoOp:
  case CK::CK_ArrayToPointerDecay:
  case CK::CK_FunctionToPointerDecay:
    return generateExpr(operand);
  case CK::CK_ToVoid:
    (void)emitDiscarded(*operand);
    return {};
  default:
    break;
  }

  auto value = generateExpr(operand);
  if (!value) {
    return {};
  }
  auto target = expr->getType()->isPointerType()
                    ? types.convertStorageType(expr->getType())
                    : types.convertType(expr->getType());
  if (!target) {
    return {};
  }
  bool isSigned = operand->getType()->isSignedIntegerOrEnumerationType();
  switch (expr->getCastKind()) {
  case CK::CK_IntegralToFloating:
  case CK::CK_IntegralCast:
  case CK::CK_FloatingCast:
  case CK::CK_BooleanToSignedIntegral:
  case CK::CK_IntegralToPointer:
  case CK::CK_PointerToIntegral:
    return clang_mlir::toValue(builder, loc, value, target, isSigned);
  case CK::CK_FloatingToIntegral:
    return clang_mlir::toValue(
        builder, loc, value, target,
        expr->getType()->isSignedIntegerOrEnumerationType());
  case CK::CK_IntegralToBoolean:
  case CK::CK_FloatingToBoolean:
  case CK::CK_PointerToBoolean:
    return clang_mlir::toBool(builder, loc, value);
  case CK::CK_NullToPointer:
    return clang_mlir::toNullPointer(builder, loc, target);
  case CK::CK_BitCast:
  case CK::CK_LValueBitCast:
    return clang_mlir::toBitcastValue(builder, loc, value, target, isSigned);
  default:
    context.source().error(expr->getExprLoc(),
                           "unsupported cast kind: " +
                               llvm::Twine(expr->getCastKindName()).str());
    return {};
  }
}

} // namespace cmlir::frontend
