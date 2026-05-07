#include "../../Converter.h"
#include "../Utils/Cast.h"
#include "llvm/Support/WithColor.h"

namespace chwc {

auto CHWConverter::generateImplicitCastExpr(clang::ImplicitCastExpr *castExpr)
    -> mlir::Value {
  mlir::OpBuilder &builder = contextManager.Builder();
  mlir::Location loc = builder.getUnknownLoc();

  clang::Expr *subExpr = castExpr->getSubExpr();
  mlir::Type targetType = convertType(castExpr->getType());

  using CK = clang::CastKind;

  switch (castExpr->getCastKind()) {
  case CK::CK_LValueToRValue:
  case CK::CK_NoOp:
  case CK::CK_ConstructorConversion:
  case CK::CK_UserDefinedConversion: {
    mlir::Value subValue = generateExpr(subExpr);
    if (!subValue) {
      return nullptr;
    }

    if (!targetType) {
      return subValue;
    }

    return utils::promoteValue(builder, loc, subValue, targetType);
  }

  case CK::CK_IntegralCast: {
    mlir::Value subValue = generateExpr(subExpr);
    if (!subValue) {
      return nullptr;
    }

    if (!targetType) {
      return subValue;
    }

    return utils::promoteValue(builder, loc, subValue, targetType);
  }

  case CK::CK_IntegralToBoolean: {
    mlir::Value subValue = generateExpr(subExpr);
    if (!subValue) {
      return nullptr;
    }

    return utils::toBool(builder, loc, subValue);
  }

  case CK::CK_BooleanToSignedIntegral: {
    mlir::Value subValue = generateExpr(subExpr);
    if (!subValue) {
      return nullptr;
    }

    if (!targetType) {
      return subValue;
    }

    return utils::promoteValue(builder, loc, subValue, targetType);
  }

  default: {
    mlir::Value subValue = generateExpr(subExpr);
    if (!subValue) {
      return nullptr;
    }

    llvm::WithColor::error()
        << "chwc: unsupported implicit cast kind: "
        << clang::ImplicitCastExpr::getCastKindName(castExpr->getCastKind())
        << "\n";

    if (!targetType) {
      return subValue;
    }

    return utils::promoteValue(builder, loc, subValue, targetType);
  }
  }
}

} // namespace chwc
