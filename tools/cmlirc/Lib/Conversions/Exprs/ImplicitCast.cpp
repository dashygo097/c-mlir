#include "../../Converter.h"
#include "../Utils/Constants.h"
#include "llvm/Support/WithColor.h"

namespace cmlirc {

mlir::Value
CMLIRConverter::generateImplicitCastExpr(clang::ImplicitCastExpr *castExpr) {
  mlir::OpBuilder &builder = contextManager.Builder();
  mlir::Location loc = builder.getUnknownLoc();
  clang::Expr *subExpr = castExpr->getSubExpr();

  mlir::Type targetType = convertType(castExpr->getType());

  using CK = clang::CastKind;

  switch (castExpr->getCastKind()) {
  case CK::CK_LValueToRValue: {
    mlir::Value subValue = generateExpr(subExpr);
    if (!subValue)
      return nullptr;

    if (subExpr->getType()->isPointerType())
      return subValue;

    if (auto memrefType =
            mlir::dyn_cast<mlir::MemRefType>(subValue.getType())) {
      if (memrefType.hasRank() && memrefType.getRank() == 0) {
        return mlir::memref::LoadOp::create(builder, loc, subValue).getResult();
      } else if (memrefType.hasRank() && memrefType.getRank() > 0) {
        if (lastArrayAccess && lastArrayAccess->base == subValue) {
          mlir::Value result =
              mlir::memref::LoadOp::create(builder, loc, lastArrayAccess->base,
                                           lastArrayAccess->indices)
                  .getResult();
          lastArrayAccess.reset();
          return result;
        } else {
          return subValue;
        }
      } else {
        return subValue;
      }
    } else if (mlir::isa<mlir::UnrankedMemRefType>(subValue.getType())) {
      return subValue;
    }

    return subValue;
  }

  case CK::CK_IntegralToFloating: {
    mlir::Value subValue = generateExpr(subExpr);
    if (!subValue)
      return nullptr;
    bool isSigned = subExpr->getType()->isSignedIntegerType();
    if (isSigned)
      return mlir::arith::SIToFPOp::create(builder, loc, targetType, subValue)
          .getResult();
    else
      return mlir::arith::UIToFPOp::create(builder, loc, targetType, subValue)
          .getResult();
  }

  case CK::CK_FloatingToIntegral: {
    mlir::Value subValue = generateExpr(subExpr);
    if (!subValue)
      return nullptr;
    bool isSigned = castExpr->getType()->isSignedIntegerType();
    if (isSigned)
      return mlir::arith::FPToSIOp::create(builder, loc, targetType, subValue)
          .getResult();
    else
      return mlir::arith::FPToUIOp::create(builder, loc, targetType, subValue)
          .getResult();
  }

  case CK::CK_IntegralCast: {
    mlir::Value subValue = generateExpr(subExpr);
    if (!subValue)
      return nullptr;
    auto srcIntType = mlir::dyn_cast<mlir::IntegerType>(subValue.getType());
    auto dstIntType = mlir::dyn_cast<mlir::IntegerType>(targetType);

    if (!srcIntType || !dstIntType)
      return subValue;

    uint32_t srcWidth = srcIntType.getWidth();
    uint32_t dstWidth = dstIntType.getWidth();

    if (srcWidth < dstWidth) {
      bool isSigned = subExpr->getType()->isSignedIntegerType();
      if (isSigned)
        return mlir::arith::ExtSIOp::create(builder, loc, targetType, subValue)
            .getResult();
      else
        return mlir::arith::ExtUIOp::create(builder, loc, targetType, subValue)
            .getResult();
    } else if (srcWidth > dstWidth) {
      return mlir::arith::TruncIOp::create(builder, loc, targetType, subValue)
          .getResult();
    }
    return subValue;
  }

  case CK::CK_FloatingCast: {
    mlir::Value subValue = generateExpr(subExpr);
    if (!subValue)
      return nullptr;
    auto srcFloatType = mlir::dyn_cast<mlir::FloatType>(subValue.getType());
    auto dstFloatType = mlir::dyn_cast<mlir::FloatType>(targetType);

    if (!srcFloatType || !dstFloatType)
      return subValue;

    if (srcFloatType.getWidth() < dstFloatType.getWidth())
      return mlir::arith::ExtFOp::create(builder, loc, targetType, subValue)
          .getResult();
    else if (srcFloatType.getWidth() > dstFloatType.getWidth())
      return mlir::arith::TruncFOp::create(builder, loc, targetType, subValue)
          .getResult();
    return subValue;
  }

  case CK::CK_IntegralToBoolean: {
    mlir::Value subValue = generateExpr(subExpr);
    if (!subValue)
      return nullptr;
    mlir::Value zero = detail::intConst(builder, loc, subValue.getType(), 0);
    return mlir::arith::CmpIOp::create(
               builder, loc, mlir::arith::CmpIPredicate::ne, subValue, zero)
        .getResult();
  }

  case CK::CK_FloatingToBoolean: {
    mlir::Value subValue = generateExpr(subExpr);
    if (!subValue)
      return nullptr;
    mlir::Value zero =
        detail::floatConst(builder, loc, subValue.getType(), 0.0);
    return mlir::arith::CmpFOp::create(
               builder, loc, mlir::arith::CmpFPredicate::UNE, subValue, zero)
        .getResult();
  }

  case CK::CK_BooleanToSignedIntegral: {
    mlir::Value subValue = generateExpr(subExpr);
    if (!subValue)
      return nullptr;
    return mlir::arith::ExtUIOp::create(builder, loc, targetType, subValue)
        .getResult();
  }

  case CK::CK_BitCast: {
    mlir::Value subValue = generateExpr(subExpr);
    if (!subValue)
      return nullptr;
    return mlir::arith::BitcastOp::create(builder, loc, targetType, subValue)
        .getResult();
  }

  case CK::CK_NoOp:
  case CK::CK_ArrayToPointerDecay:
  case CK::CK_FunctionToPointerDecay: {
    mlir::Value subValue = generateExpr(subExpr);
    if (!subValue)
      return nullptr;
    return subValue;
  }

  default: {
    mlir::Value subValue = generateExpr(subExpr);
    if (!subValue)
      return nullptr;
    llvm::WithColor::error()
        << "cmlirc: unsupported cast kind: "
        << clang::ImplicitCastExpr::getCastKindName(castExpr->getCastKind())
        << "\n";
    return subValue;
  }
  }
}

} // namespace cmlirc
