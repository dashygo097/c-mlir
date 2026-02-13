#include "../../Converter.h"
#include "../Types/Types.h"

namespace cmlirc {

mlir::Value
CMLIRConverter::generateImplicitCastExpr(clang::ImplicitCastExpr *castExpr) {
  mlir::OpBuilder &builder = context_manager_.Builder();
  mlir::Location loc = builder.getUnknownLoc();
  clang::CastKind castKind = castExpr->getCastKind();

  mlir::Value subValue = generateExpr(castExpr->getSubExpr());
  if (!subValue) {
    return nullptr;
  }

  mlir::Type targetType = convertType(builder, castExpr->getType());

  switch (castKind) {
  case clang::CK_LValueToRValue: {
    if (auto memrefType =
            mlir::dyn_cast<mlir::MemRefType>(subValue.getType())) {
      if (memrefType.hasRank() && memrefType.getRank() == 0) {
        return mlir::memref::LoadOp::create(builder, loc, subValue).getResult();
      } else if (memrefType.hasRank() && memrefType.getRank() > 0) {
        if (lastArrayAccess_ && lastArrayAccess_->base == subValue) {
          mlir::Value result =
              mlir::memref::LoadOp::create(builder, loc, lastArrayAccess_->base,
                                           lastArrayAccess_->indices)
                  .getResult();
          lastArrayAccess_.reset();
          return result;
        } else {
          llvm::errs() << "Loading array without indices\n";
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

  case clang::CK_IntegralToFloating: {
    bool isSigned = castExpr->getSubExpr()->getType()->isSignedIntegerType();
    if (isSigned) {
      return mlir::arith::SIToFPOp::create(builder, loc, targetType, subValue)
          .getResult();
    } else {
      return mlir::arith::UIToFPOp::create(builder, loc, targetType, subValue)
          .getResult();
    }
  }

  case clang::CK_FloatingToIntegral: {
    bool isSigned = castExpr->getType()->isSignedIntegerType();
    if (isSigned) {
      return mlir::arith::FPToSIOp::create(builder, loc, targetType, subValue)
          .getResult();
    } else {
      return mlir::arith::FPToUIOp::create(builder, loc, targetType, subValue)
          .getResult();
    }
  }

  case clang::CK_IntegralCast: {
    auto srcIntType = mlir::dyn_cast<mlir::IntegerType>(subValue.getType());
    auto dstIntType = mlir::dyn_cast<mlir::IntegerType>(targetType);

    if (!srcIntType || !dstIntType) {
      return subValue;
    }

    unsigned srcWidth = srcIntType.getWidth();
    unsigned dstWidth = dstIntType.getWidth();

    if (srcWidth < dstWidth) {
      bool isSigned = castExpr->getSubExpr()->getType()->isSignedIntegerType();
      if (isSigned) {
        return mlir::arith::ExtSIOp::create(builder, loc, targetType, subValue)
            .getResult();
      } else {
        return mlir::arith::ExtUIOp::create(builder, loc, targetType, subValue)
            .getResult();
      }
    } else if (srcWidth > dstWidth) {
      return mlir::arith::TruncIOp::create(builder, loc, targetType, subValue)
          .getResult();
    }

    return subValue;
  }

  case clang::CK_FloatingCast: {
    auto srcFloatType = mlir::dyn_cast<mlir::FloatType>(subValue.getType());
    auto dstFloatType = mlir::dyn_cast<mlir::FloatType>(targetType);

    if (!srcFloatType || !dstFloatType) {
      return subValue;
    }

    if (srcFloatType.getWidth() < dstFloatType.getWidth()) {
      return mlir::arith::ExtFOp::create(builder, loc, targetType, subValue)
          .getResult();
    } else if (srcFloatType.getWidth() > dstFloatType.getWidth()) {
      return mlir::arith::TruncFOp::create(builder, loc, targetType, subValue)
          .getResult();
    }

    return subValue;
  }

  case clang::CK_IntegralToBoolean: {
    auto zeroAttr = builder.getIntegerAttr(subValue.getType(), 0);
    mlir::Value zero =
        mlir::arith::ConstantOp::create(builder, loc, zeroAttr).getResult();
    return mlir::arith::CmpIOp::create(
               builder, loc, mlir::arith::CmpIPredicate::ne, subValue, zero)
        .getResult();
  }

  case clang::CK_FloatingToBoolean: {
    auto zeroAttr = builder.getFloatAttr(subValue.getType(), 0.0);
    mlir::Value zero =
        mlir::arith::ConstantOp::create(builder, loc, zeroAttr).getResult();
    return mlir::arith::CmpFOp::create(
               builder, loc, mlir::arith::CmpFPredicate::UNE, subValue, zero)
        .getResult();
  }

  case clang::CK_BooleanToSignedIntegral: {
    return mlir::arith::ExtUIOp::create(builder, loc, targetType, subValue)
        .getResult();
  }

  case clang::CK_BitCast: {
    return mlir::arith::BitcastOp::create(builder, loc, targetType, subValue)
        .getResult();
  }

  case clang::CK_NoOp:
  case clang::CK_ArrayToPointerDecay:
  case clang::CK_FunctionToPointerDecay:
    return subValue;

  default:
    llvm::errs() << "Unsupported cast kind: "
                 << clang::ImplicitCastExpr::getCastKindName(castKind) << "\n";
    return subValue;
  }
}
} // namespace cmlirc
