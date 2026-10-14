#include "../Lowering.h"
#include "clang-mlir/Casts.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "llvm/Support/WithColor.h"

namespace cmlir::frontend {

auto ASTLowering::generateImplicitCastExpr(clang::ImplicitCastExpr *castExpr)
    -> mlir::Value {
  mlir::OpBuilder &builder = context.builder();
  mlir::Location loc = context.source().location(castExpr->getExprLoc());
  clang::Expr *subExpr = castExpr->getSubExpr();

  mlir::Type targetType = convertType(castExpr->getType());

  using CK = clang::CastKind;

  switch (castExpr->getCastKind()) {
  case clang::CK_LValueToRValue:
    return loadLValue(subExpr);

  case CK::CK_IntegralToFloating:
  case CK::CK_IntegralCast:
  case CK::CK_FloatingCast:
  case CK::CK_BooleanToSignedIntegral: {
    mlir::Value value = generateExpr(subExpr);
    bool isSigned = subExpr->getType()->isSignedIntegerOrEnumerationType();
    return clang_mlir::toValue(builder, loc, value, targetType, isSigned);
  }

  case CK::CK_FloatingToIntegral: {
    mlir::Value value = generateExpr(subExpr);
    bool isSigned = castExpr->getType()->isSignedIntegerOrEnumerationType();
    return clang_mlir::toValue(builder, loc, value, targetType, isSigned);
  }

  case CK::CK_IntegralToBoolean:
  case CK::CK_FloatingToBoolean:
  case CK::CK_PointerToBoolean: {
    mlir::Value value = generateExpr(subExpr);
    return clang_mlir::toBool(builder, loc, value);
  }

  case CK::CK_IntegralToPointer: {
    mlir::Value value = generateExpr(subExpr);
    return clang_mlir::toPointer(builder, loc, value, targetType);
  }

  case CK::CK_PointerToIntegral: {
    mlir::Value value = generateExpr(subExpr);
    bool isSigned = castExpr->getType()->isSignedIntegerOrEnumerationType();
    return clang_mlir::toValue(builder, loc, value, targetType, isSigned);
  }

  case CK::CK_NullToPointer: {
    return clang_mlir::toNullPointer(builder, loc, targetType);
  }

  case CK::CK_BitCast:
  case CK::CK_LValueBitCast:
  case CK::CK_AddressSpaceConversion: {
    mlir::Value value = generateExpr(subExpr);
    bool isSigned = subExpr->getType()->isSignedIntegerOrEnumerationType();

    if (castExpr->getType()->isPointerType() &&
        mlir::isa<mlir::LLVM::LLVMPointerType>(value.getType())) {
      return value;
    }

    return clang_mlir::toBitcastValue(builder, loc, value, targetType,
                                      isSigned);
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
        << "clang-to-mlir: unsupported cast kind: "
        << clang::ImplicitCastExpr::getCastKindName(castExpr->getCastKind())
        << "\n";
    return nullptr;
  }
  }
}

} // namespace cmlir::frontend
