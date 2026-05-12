#include "../../Converter.h"
#include "../Utils/Casts.h"
#include "../Utils/Constants.h"
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
    mlir::Value subValue = generateExpr(subExpr);

    if (mlir::isa<mlir::LLVM::LLVMPointerType>(subValue.getType()) &&
        !lastArrayAccess) {
      return mlir::LLVM::LoadOp::create(builder, loc, targetType, subValue)
          .getResult();
    }

    if (subExpr->getType()->isPointerType() && !lastArrayAccess) {
      return subValue;
    }

    if (lastArrayAccess && lastArrayAccess->base == subValue) {

      if (mlir::isa<mlir::LLVM::LLVMPointerType>(subValue.getType())) {
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

    if (auto memrefType =
            mlir::dyn_cast<mlir::MemRefType>(subValue.getType())) {
      if (memrefType.hasRank() && memrefType.getRank() == 0) {
        return mlir::memref::LoadOp::create(builder, loc, subValue).getResult();
      }
    }
    return subValue;
  }

  case CK::CK_IntegralToFloating:
  case CK::CK_FloatingToIntegral:
  case CK::CK_IntegralCast:
  case CK::CK_FloatingCast:
  case CK::CK_BooleanToSignedIntegral: {
    mlir::Value subValue = generateExpr(subExpr);
    bool isSigned = subExpr->getType()->isSignedIntegerType();
    return utils::castValue(builder, loc, subValue, targetType, isSigned);
  }

  case CK::CK_IntegralToBoolean:
  case CK::CK_FloatingToBoolean: {
    mlir::Value subValue = generateExpr(subExpr);
    return utils::toBool(builder, loc, subValue);
  }

  case CK::CK_BitCast: {
    mlir::Value subValue = generateExpr(subExpr);

    if (mlir::isa<mlir::MemRefType>(subValue.getType()) &&
        mlir::isa<mlir::LLVM::LLVMPointerType>(targetType)) {

      mlir::Value ptrAsIndex =
          mlir::memref::ExtractAlignedPointerAsIndexOp::create(
              builder, loc, builder.getIndexType(), subValue);

      mlir::Value ptrAsI64 = mlir::arith::IndexCastOp::create(
          builder, loc, builder.getI64Type(), ptrAsIndex);

      return mlir::LLVM::IntToPtrOp::create(builder, loc, targetType, ptrAsI64)
          .getResult();
    }

    return mlir::arith::BitcastOp::create(builder, loc, targetType, subValue)
        .getResult();
  }

  case CK::CK_NoOp:
  case CK::CK_ArrayToPointerDecay:
  case CK::CK_FunctionToPointerDecay: {
    mlir::Value subValue = generateExpr(subExpr);
    return subValue;
  }

  default: {
    mlir::Value subValue = generateExpr(subExpr);
    llvm::WithColor::error()
        << "cmlirc: unsupported cast kind: "
        << clang::ImplicitCastExpr::getCastKindName(castExpr->getCastKind())
        << "\n";
    return subValue;
  }
  }
}

} // namespace cmlirc
