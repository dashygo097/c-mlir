#include "../../Converter.h"
#include "../Utils/Casts.h"
#include "llvm/Support/WithColor.h"

namespace cmlirc {

auto CMLIRConverter::generateCStyleCastExpr(clang::CStyleCastExpr *castExpr)
    -> mlir::Value {
  mlir::OpBuilder &builder = contextManager.Builder();
  mlir::Location loc = builder.getUnknownLoc();
  clang::Expr *subExpr = castExpr->getSubExpr();

  mlir::Value value = generateExpr(subExpr);
  mlir::Type targetType = convertType(castExpr->getType());

  switch (castExpr->getCastKind()) {
  case clang::CK_IntegralCast:
  case clang::CK_FloatingCast:
  case clang::CK_BooleanToSignedIntegral:
  case clang::CK_IntegralToFloating: {
    bool isSigned = subExpr->getType()->isSignedIntegerOrEnumerationType();
    return utils::toValue(builder, loc, value, targetType, isSigned);
  }

  case clang::CK_FloatingToIntegral: {
    bool isSigned = castExpr->getType()->isSignedIntegerOrEnumerationType();
    return utils::toValue(builder, loc, value, targetType, isSigned);
  }

  case clang::CK_IntegralToBoolean:
  case clang::CK_FloatingToBoolean:
  case clang::CK_PointerToBoolean: {
    return utils::toBool(builder, loc, value);
  }

  case clang::CK_IntegralToPointer: {
    return utils::toPointer(builder, loc, value, targetType);
  }

  case clang::CK_PointerToIntegral: {
    bool isSigned = castExpr->getType()->isSignedIntegerOrEnumerationType();
    return utils::toValue(builder, loc, value, targetType, isSigned);
  }

  case clang::CK_NullToPointer: {
    return utils::toNullPointer(builder, loc, targetType);
  }

  case clang::CK_BitCast:
  case clang::CK_LValueBitCast:
  case clang::CK_AddressSpaceConversion: {
    if (castExpr->getType()->isPointerType() &&
        mlir::isa<mlir::LLVM::LLVMPointerType>(value.getType())) {
      return value;
    }

    bool isSigned = subExpr->getType()->isSignedIntegerOrEnumerationType();
    return utils::toBitcastValue(builder, loc, value, targetType, isSigned);
  }

  case clang::CK_NoOp: {
    if (castExpr->getType()->isPointerType() &&
        mlir::isa<mlir::LLVM::LLVMPointerType>(value.getType())) {
      return value;
    }

    return value;
  }

  default:
    llvm::WithColor::error()
        << "cmlirc: unsupported C-style cast kind: "
        << clang::CStyleCastExpr::getCastKindName(castExpr->getCastKind())
        << "\n";
    return nullptr;
  }
}

} // namespace cmlirc
