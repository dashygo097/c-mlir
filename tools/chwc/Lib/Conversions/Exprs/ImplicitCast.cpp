#include "../../Converter.h"
#include "../Utils/Cast.h"
#include "../Utils/Comb.h"
#include "clang/AST/OperationKinds.h"
#include "llvm/Support/WithColor.h"

namespace chwc {

auto CHWConverter::generateImplicitCastExpr(clang::ImplicitCastExpr *castExpr)
    -> mlir::Value {
  clang::Expr *subExpr = castExpr->getSubExpr();

  switch (castExpr->getCastKind()) {
  case clang::CK_NoOp:
  case clang::CK_LValueToRValue:
  case clang::CK_UserDefinedConversion:
  case clang::CK_ConstructorConversion:
    return generateExpr(subExpr);

  case clang::CK_IntegralCast:
  case clang::CK_IntegralToBoolean:
  case clang::CK_BooleanToSignedIntegral: {
    mlir::Value value = generateExpr(subExpr);
    if (!value) {
      return nullptr;
    }

    mlir::Type targetType = convertType(castExpr->getType());
    if (!targetType) {
      return value;
    }

    mlir::OpBuilder &builder = contextManager.Builder();
    mlir::Location loc = builder.getUnknownLoc();

    return utils::promoteValue(builder, loc, value, targetType);
  }

  default:
    llvm::WithColor::error()
        << "chwc: unsupported implicit cast kind: "
        << clang::ImplicitCastExpr::getCastKindName(castExpr->getCastKind())
        << "\n";
    return generateExpr(subExpr);
  }
}

} // namespace chwc
