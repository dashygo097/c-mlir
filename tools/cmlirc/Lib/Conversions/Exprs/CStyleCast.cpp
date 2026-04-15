#include "../../Converter.h"
#include "../Utils/Casts.h"
#include "llvm/Support/WithColor.h"

namespace cmlirc {

auto CMLIRConverter::generateCStyleCastExpr(clang::CStyleCastExpr *castExpr)
    -> mlir::Value {
  mlir::OpBuilder &builder = contextManager.Builder();
  mlir::Location loc = builder.getUnknownLoc();
  clang::Expr *subExpr = castExpr->getSubExpr();

  mlir::Value subValue = generateExpr(subExpr);
  if (!subValue) {
    return nullptr;
  }

  mlir::Type targetType = convertType(castExpr->getType());

  switch (castExpr->getCastKind()) {

  // Source-dependent Sign Casts (Int -> Float, Int -> Wider Int)
  case clang::CK_IntegralToFloating:
  case clang::CK_IntegralCast:
  case clang::CK_FloatingCast:
  case clang::CK_BooleanToSignedIntegral: {
    bool isSigned = subExpr->getType()->isSignedIntegerType();
    return utils::castValue(builder, loc, subValue, targetType, isSigned);
  }

  // Target-dependent Sign Casts (Float -> Int)
  case clang::CK_FloatingToIntegral: {
    bool isSigned = castExpr->getType()->isSignedIntegerType();
    return utils::castValue(builder, loc, subValue, targetType, isSigned);
  }

  // Boolean Casts (Evaluates against 0 / 0.0)
  case clang::CK_IntegralToBoolean:
  case clang::CK_FloatingToBoolean: {
    return utils::toBool(builder, loc, subValue);
  }

  // Memory/Bitwise Casts
  case clang::CK_BitCast: {
    return mlir::arith::BitcastOp::create(builder, loc, targetType, subValue)
        .getResult();
  }

  case clang::CK_NoOp: {
    return subValue;
  }

  default:
    llvm::WithColor::error()
        << "cmlirc: unsupported C-style cast kind: "
        << clang::CStyleCastExpr::getCastKindName(castExpr->getCastKind())
        << "\n";
    return subValue;
  }
}

} // namespace cmlirc
