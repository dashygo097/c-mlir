#include "../../Converter.h"
#include "../Utils/Casts.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
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
  case CK::CK_UserDefinedConversion: {
    mlir::Value subValue = generateExpr(subExpr);
    if (!subValue) {
      return nullptr;
    }
    return subValue;
  }

  case CK::CK_IntegralCast: {
    mlir::Value subValue = generateExpr(subExpr);
    if (!subValue) {
      return nullptr;
    }

    auto srcIntType = mlir::dyn_cast<mlir::IntegerType>(subValue.getType());
    auto dstIntType = mlir::dyn_cast<mlir::IntegerType>(targetType);

    if (!srcIntType || !dstIntType) {
      return subValue;
    }

    uint32_t srcWidth = srcIntType.getWidth();
    uint32_t dstWidth = dstIntType.getWidth();

    if (srcWidth < dstWidth) {
      bool isSigned = subExpr->getType()->isSignedIntegerType();
      if (isSigned) {
        return mlir::arith::ExtSIOp::create(builder, loc, targetType, subValue)
            .getResult();
      }

      return mlir::arith::ExtUIOp::create(builder, loc, targetType, subValue)
          .getResult();
    }

    if (srcWidth > dstWidth) {
      return mlir::arith::TruncIOp::create(builder, loc, targetType, subValue)
          .getResult();
    }

    return subValue;
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

    auto srcIntType = mlir::dyn_cast<mlir::IntegerType>(subValue.getType());
    auto dstIntType = mlir::dyn_cast<mlir::IntegerType>(targetType);

    if (!srcIntType || !dstIntType) {
      return subValue;
    }

    if (srcIntType.getWidth() < dstIntType.getWidth()) {
      return mlir::arith::ExtUIOp::create(builder, loc, targetType, subValue)
          .getResult();
    }

    if (srcIntType.getWidth() > dstIntType.getWidth()) {
      return mlir::arith::TruncIOp::create(builder, loc, targetType, subValue)
          .getResult();
    }

    return subValue;
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

    return subValue;
  }
  }
}

} // namespace chwc
