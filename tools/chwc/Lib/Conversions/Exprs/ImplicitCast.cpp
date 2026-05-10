#include "../../Converter.h"
#include "../Utils/Cast.h"
#include "clang/AST/OperationKinds.h"
#include "llvm/Support/WithColor.h"

namespace chwc {

auto CHWConverter::generateImplicitCastExpr(clang::ImplicitCastExpr *castExpr)
    -> mlir::Value {
  if (!castExpr) {
    return nullptr;
  }

  clang::Expr *subExpr = castExpr->getSubExpr();
  if (!subExpr) {
    return nullptr;
  }

  switch (castExpr->getCastKind()) {
  case clang::CK_NoOp:
  case clang::CK_LValueToRValue:
  case clang::CK_DerivedToBase:
  case clang::CK_UncheckedDerivedToBase:
  case clang::CK_UserDefinedConversion:
    return generateExpr(subExpr);

  case clang::CK_IntegralCast:
  case clang::CK_IntegralToBoolean:
  case clang::CK_BooleanToSignedIntegral:
  case clang::CK_IntegralToFloating:
  case clang::CK_FloatingToIntegral: {
    mlir::Value value = generateExpr(subExpr);
    if (!value) {
      return nullptr;
    }

    mlir::Type targetType = convertType(castExpr->getType());
    if (!targetType) {
      return value;
    }

    if (value.getType() == targetType) {
      return value;
    }

    mlir::OpBuilder &builder = contextManager.Builder();
    mlir::Location loc = builder.getUnknownLoc();

    return utils::promoteValue(builder, loc, value, targetType);
  }

  default:
    llvm::WithColor::error() << "chwc: unsupported implicit cast kind: "
                             << castExpr->getCastKindName() << "\n";
    return nullptr;
  }
}

} // namespace chwc
