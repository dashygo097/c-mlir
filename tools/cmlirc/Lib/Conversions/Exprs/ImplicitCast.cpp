#include "../../Converter.h"
#include "../Utils/Casts.h"
#include "../Utils/LHS.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "llvm/Support/WithColor.h"

namespace cmlirc {

auto CMLIRConverter::generateImplicitCastExpr(clang::ImplicitCastExpr *castExpr)
    -> mlir::Value {
  mlir::OpBuilder &builder = contextManager.Builder();
  mlir::Location loc = builder.getUnknownLoc();
  clang::Expr *subExpr = castExpr->getSubExpr();

  mlir::Type targetType = convertType(castExpr->getType());

  using CK = clang::CastKind;

  switch (castExpr->getCastKind()) {
  case clang::CK_LValueToRValue: {
    mlir::Value value = generateExpr(subExpr);

    if (mlir::isa<mlir::LLVM::LLVMPointerType>(value.getType()) &&
        !lastArrayAccess) {
      return mlir::LLVM::LoadOp::create(builder, loc, targetType, value)
          .getResult();
    }

    if (subExpr->getType()->isPointerType() && !lastArrayAccess) {
      return value;
    }

    if (lastArrayAccess && lastArrayAccess->base == value) {
      if (mlir::isa<mlir::LLVM::LLVMPointerType>(value.getType())) {
        mlir::Value offsetPtr =
            utils::getLLVMOffsetPointer(builder, loc, lastArrayAccess->base,
                                        targetType, lastArrayAccess->indices);

        mlir::Value result =
            mlir::LLVM::LoadOp::create(builder, loc, targetType, offsetPtr)
                .getResult();

        lastArrayAccess.reset();
        return result;
      }

      mlir::Value result =
          mlir::memref::LoadOp::create(builder, loc, lastArrayAccess->base,
                                       lastArrayAccess->indices)
              .getResult();

      lastArrayAccess.reset();
      return result;
    }

    if (auto memrefType = mlir::dyn_cast<mlir::MemRefType>(value.getType())) {
      if (memrefType.hasRank() && memrefType.getRank() == 0) {
        return mlir::memref::LoadOp::create(builder, loc, value).getResult();
      }
    }

    return value;
  }

  case CK::CK_IntegralToFloating:
  case CK::CK_IntegralCast:
  case CK::CK_FloatingCast:
  case CK::CK_BooleanToSignedIntegral: {
    mlir::Value value = generateExpr(subExpr);
    bool isSigned = subExpr->getType()->isSignedIntegerOrEnumerationType();
    return utils::toValue(builder, loc, value, targetType, isSigned);
  }

  case CK::CK_FloatingToIntegral: {
    mlir::Value value = generateExpr(subExpr);
    bool isSigned = castExpr->getType()->isSignedIntegerOrEnumerationType();
    return utils::toValue(builder, loc, value, targetType, isSigned);
  }

  case CK::CK_IntegralToBoolean:
  case CK::CK_FloatingToBoolean:
  case CK::CK_PointerToBoolean: {
    mlir::Value value = generateExpr(subExpr);
    return utils::toBool(builder, loc, value);
  }

  case CK::CK_IntegralToPointer: {
    mlir::Value value = generateExpr(subExpr);
    return utils::toPointer(builder, loc, value, targetType);
  }

  case CK::CK_PointerToIntegral: {
    mlir::Value value = generateExpr(subExpr);
    bool isSigned = castExpr->getType()->isSignedIntegerOrEnumerationType();
    return utils::toValue(builder, loc, value, targetType, isSigned);
  }

  case CK::CK_NullToPointer: {
    return utils::toNullPointer(builder, loc, targetType);
  }

  case CK::CK_BitCast:
  case CK::CK_LValueBitCast:
  case CK::CK_AddressSpaceConversion: {
    mlir::Value value = generateExpr(subExpr);
    bool isSigned = subExpr->getType()->isSignedIntegerOrEnumerationType();
    return utils::toBitcastValue(builder, loc, value, targetType, isSigned);
  }

  case CK::CK_NoOp:
  case CK::CK_ArrayToPointerDecay:
  case CK::CK_FunctionToPointerDecay: {
    return generateExpr(subExpr);
  }

  case CK::CK_ToVoid: {
    generateExpr(subExpr);
    return nullptr;
  }

  default: {
    llvm::WithColor::error()
        << "cmlirc: unsupported cast kind: "
        << clang::ImplicitCastExpr::getCastKindName(castExpr->getCastKind())
        << "\n";
    return nullptr;
  }
  }
}

} // namespace cmlirc
