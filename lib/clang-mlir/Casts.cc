#include "clang-mlir/Casts.h"
#include "clang-mlir/MemrefABI.h"

namespace clang_mlir {

auto memrefToPointer(mlir::OpBuilder &builder, mlir::Location loc,
                     mlir::Value value, mlir::Type targetType) -> mlir::Value {
  if (!value || !targetType) {
    return nullptr;
  }

  if (!mlir::isa<mlir::MemRefType>(value.getType()) ||
      !mlir::isa<mlir::LLVM::LLVMPointerType>(targetType)) {
    return nullptr;
  }

  mlir::Value ptrAsIndex = mlir::memref::ExtractAlignedPointerAsIndexOp::create(
                               builder, loc, builder.getIndexType(), value)
                               .getResult();

  mlir::Value ptrAsI64 = mlir::arith::IndexCastOp::create(
                             builder, loc, builder.getI64Type(), ptrAsIndex)
                             .getResult();

  auto pointer =
      mlir::LLVM::IntToPtrOp::create(builder, loc, targetType, ptrAsI64);
  if (mlir::cast<mlir::MemRefType>(value.getType()).getLayout().isIdentity()) {
    return pointer;
  }
  auto metadata =
      mlir::memref::ExtractStridedMetadataOp::create(builder, loc, value);
  auto offset = mlir::arith::IndexCastOp::create(
      builder, loc, builder.getI64Type(), metadata.getOffset());
  auto element = mlir::cast<mlir::MemRefType>(value.getType()).getElementType();
  return mlir::LLVM::GEPOp::create(
      builder, loc, targetType, element, pointer,
      llvm::ArrayRef<mlir::LLVM::GEPArg>{offset.getResult()});
}

auto integerToPointer(mlir::OpBuilder &builder, mlir::Location loc,
                      mlir::Value value, mlir::Type targetType) -> mlir::Value {
  if (!value || !targetType) {
    return nullptr;
  }

  if (!mlir::isa<mlir::LLVM::LLVMPointerType>(targetType)) {
    return nullptr;
  }

  mlir::Type srcType = value.getType();

  if (srcType.isIndex()) {
    value = mlir::arith::IndexCastOp::create(builder, loc, builder.getI64Type(),
                                             value)
                .getResult();
    srcType = value.getType();
  }

  auto intType = mlir::dyn_cast<mlir::IntegerType>(srcType);
  if (!intType) {
    return nullptr;
  }

  if (intType.getWidth() != 64) {
    if (intType.getWidth() < 64) {
      value = mlir::arith::ExtUIOp::create(builder, loc, builder.getI64Type(),
                                           value)
                  .getResult();
    } else {
      value = mlir::arith::TruncIOp::create(builder, loc, builder.getI64Type(),
                                            value)
                  .getResult();
    }
  }

  return mlir::LLVM::IntToPtrOp::create(builder, loc, targetType, value)
      .getResult();
}

auto pointerToInteger(mlir::OpBuilder &builder, mlir::Location loc,
                      mlir::Value value, mlir::Type targetType) -> mlir::Value {
  if (!value || !targetType) {
    return nullptr;
  }

  if (!mlir::isa<mlir::LLVM::LLVMPointerType>(value.getType())) {
    return nullptr;
  }

  if (targetType.isIndex()) {
    mlir::Value ptrAsI64 = mlir::LLVM::PtrToIntOp::create(
                               builder, loc, builder.getI64Type(), value)
                               .getResult();

    return mlir::arith::IndexCastOp::create(builder, loc,
                                            builder.getIndexType(), ptrAsI64)
        .getResult();
  }

  if (mlir::isa<mlir::IntegerType>(targetType)) {
    return mlir::LLVM::PtrToIntOp::create(builder, loc, targetType, value)
        .getResult();
  }

  return nullptr;
}

auto indexToInteger(mlir::OpBuilder &builder, mlir::Location loc,
                    mlir::Value value, mlir::IntegerType targetType)
    -> mlir::Value {
  if (!value || !targetType || !value.getType().isIndex()) {
    return nullptr;
  }

  return mlir::arith::IndexCastOp::create(builder, loc, targetType, value)
      .getResult();
}

auto integerToIndex(mlir::OpBuilder &builder, mlir::Location loc,
                    mlir::Value value) -> mlir::Value {
  if (!value || !mlir::isa<mlir::IntegerType>(value.getType())) {
    return nullptr;
  }

  return mlir::arith::IndexCastOp::create(builder, loc, builder.getIndexType(),
                                          value)
      .getResult();
}

auto indexToInteger64(mlir::OpBuilder &builder, mlir::Location loc,
                      mlir::Value value) -> mlir::Value {
  return indexToInteger(builder, loc, value, builder.getI64Type());
}

auto memrefToInteger(mlir::OpBuilder &builder, mlir::Location loc,
                     mlir::Value value, mlir::IntegerType targetType)
    -> mlir::Value {
  if (!value || !targetType || !mlir::isa<mlir::MemRefType>(value.getType())) {
    return nullptr;
  }

  auto ptrType = mlir::LLVM::LLVMPointerType::get(builder.getContext());
  mlir::Value ptr = memrefToPointer(builder, loc, value, ptrType);

  if (!ptr) {
    return nullptr;
  }

  return pointerToInteger(builder, loc, ptr, targetType);
}

auto toInteger(mlir::OpBuilder &builder, mlir::Location loc, mlir::Value value,
               mlir::IntegerType targetType, bool isSigned) -> mlir::Value {
  if (!value || !targetType) {
    return nullptr;
  }

  mlir::Type srcType = value.getType();

  if (srcType == targetType) {
    return value;
  }

  if (srcType.isIndex()) {
    return indexToInteger(builder, loc, value, targetType);
  }

  if (auto srcInt = mlir::dyn_cast<mlir::IntegerType>(srcType)) {
    if (srcInt.getWidth() < targetType.getWidth()) {
      bool useSignedExtend = isSigned && !srcType.isInteger(1);

      return useSignedExtend
                 ? mlir::arith::ExtSIOp::create(builder, loc, targetType, value)
                       .getResult()
                 : mlir::arith::ExtUIOp::create(builder, loc, targetType, value)
                       .getResult();
    }

    if (srcInt.getWidth() > targetType.getWidth()) {
      return mlir::arith::TruncIOp::create(builder, loc, targetType, value)
          .getResult();
    }

    return value;
  }

  if (mlir::isa<mlir::FloatType>(srcType)) {
    return isSigned
               ? mlir::arith::FPToSIOp::create(builder, loc, targetType, value)
                     .getResult()
               : mlir::arith::FPToUIOp::create(builder, loc, targetType, value)
                     .getResult();
  }

  if (mlir::isa<mlir::LLVM::LLVMPointerType>(srcType)) {
    return pointerToInteger(builder, loc, value, targetType);
  }

  if (mlir::isa<mlir::MemRefType>(srcType)) {
    return memrefToInteger(builder, loc, value, targetType);
  }

  return nullptr;
}

auto toIndex(mlir::OpBuilder &builder, mlir::Location loc, mlir::Value value,
             bool isSigned) -> mlir::Value {
  if (!value) {
    return nullptr;
  }

  if (value.getType().isIndex()) {
    return value;
  }

  mlir::Value integer =
      toInteger(builder, loc, value, builder.getI64Type(), isSigned);

  if (!integer) {
    return nullptr;
  }

  return integerToIndex(builder, loc, integer);
}

auto integerToFloat(mlir::OpBuilder &builder, mlir::Location loc,
                    mlir::Value value, mlir::FloatType targetType,
                    bool isSigned) -> mlir::Value {
  if (!value || !targetType) {
    return nullptr;
  }

  if (!mlir::isa<mlir::IntegerType>(value.getType())) {
    return nullptr;
  }

  return isSigned
             ? mlir::arith::SIToFPOp::create(builder, loc, targetType, value)
                   .getResult()
             : mlir::arith::UIToFPOp::create(builder, loc, targetType, value)
                   .getResult();
}

auto indexToFloat(mlir::OpBuilder &builder, mlir::Location loc,
                  mlir::Value value, mlir::FloatType targetType, bool isSigned)
    -> mlir::Value {
  mlir::Value integer = indexToInteger64(builder, loc, value);

  if (!integer) {
    return nullptr;
  }

  return integerToFloat(builder, loc, integer, targetType, isSigned);
}

auto floatToFloat(mlir::OpBuilder &builder, mlir::Location loc,
                  mlir::Value value, mlir::FloatType targetType)
    -> mlir::Value {
  if (!value || !targetType) {
    return nullptr;
  }

  auto srcFloat = mlir::dyn_cast<mlir::FloatType>(value.getType());
  if (!srcFloat) {
    return nullptr;
  }

  if (srcFloat == targetType) {
    return value;
  }

  if (srcFloat.getWidth() < targetType.getWidth()) {
    return mlir::arith::ExtFOp::create(builder, loc, targetType, value, {})
        .getResult();
  }

  if (srcFloat.getWidth() > targetType.getWidth()) {
    return mlir::arith::TruncFOp::create(builder, loc, targetType, value)
        .getResult();
  }

  return value;
}

auto toFloat(mlir::OpBuilder &builder, mlir::Location loc, mlir::Value value,
             mlir::FloatType targetType, bool isSigned) -> mlir::Value {
  if (!value || !targetType) {
    return nullptr;
  }

  mlir::Type srcType = value.getType();

  if (srcType == targetType) {
    return value;
  }

  if (srcType.isIndex()) {
    return indexToFloat(builder, loc, value, targetType, isSigned);
  }

  if (mlir::isa<mlir::IntegerType>(srcType)) {
    return integerToFloat(builder, loc, value, targetType, isSigned);
  }

  if (mlir::isa<mlir::FloatType>(srcType)) {
    return floatToFloat(builder, loc, value, targetType);
  }

  return nullptr;
}

auto toPointer(mlir::OpBuilder &builder, mlir::Location loc, mlir::Value value,
               mlir::Type targetType) -> mlir::Value {
  if (!value || !targetType) {
    return nullptr;
  }

  if (!mlir::isa<mlir::LLVM::LLVMPointerType>(targetType)) {
    return nullptr;
  }

  mlir::Type srcType = value.getType();

  if (srcType == targetType) {
    return value;
  }

  if (mlir::isa<mlir::LLVM::LLVMPointerType>(srcType)) {
    return value;
  }

  if (mlir::isa<mlir::MemRefType>(srcType)) {
    return memrefToPointer(builder, loc, value, targetType);
  }

  if (srcType.isIndex() || mlir::isa<mlir::IntegerType>(srcType)) {
    return integerToPointer(builder, loc, value, targetType);
  }

  return nullptr;
}

auto toNullPointer(mlir::OpBuilder &builder, mlir::Location loc,
                   mlir::Type targetType) -> mlir::Value {
  if (!targetType || !mlir::isa<mlir::LLVM::LLVMPointerType>(targetType)) {
    return nullptr;
  }

  mlir::Value zero = intConst(builder, loc, builder.getI64Type(), 0);

  return integerToPointer(builder, loc, zero, targetType);
}

auto integerToBool(mlir::OpBuilder &builder, mlir::Location loc,
                   mlir::Value value) -> mlir::Value {
  if (!value || !isIntegerLikeType(value.getType())) {
    return nullptr;
  }

  return mlir::arith::CmpIOp::create(builder, loc,
                                     mlir::arith::CmpIPredicate::ne, value,
                                     zeroConst(builder, loc, value.getType()))
      .getResult();
}

auto floatToBool(mlir::OpBuilder &builder, mlir::Location loc,
                 mlir::Value value) -> mlir::Value {
  if (!value || !isFloatType(value.getType())) {
    return nullptr;
  }

  return mlir::arith::CmpFOp::create(builder, loc,
                                     mlir::arith::CmpFPredicate::UNE, value,
                                     zeroConst(builder, loc, value.getType()))
      .getResult();
}

auto pointerToBool(mlir::OpBuilder &builder, mlir::Location loc,
                   mlir::Value value) -> mlir::Value {
  if (!value || !mlir::isa<mlir::LLVM::LLVMPointerType>(value.getType())) {
    return nullptr;
  }

  mlir::Value ptrAsInt =
      pointerToInteger(builder, loc, value, builder.getI64Type());

  if (!ptrAsInt) {
    return nullptr;
  }

  return integerToBool(builder, loc, ptrAsInt);
}

auto memrefToBool(mlir::OpBuilder &builder, mlir::Location loc,
                  mlir::Value value) -> mlir::Value {
  if (!value || !mlir::isa<mlir::MemRefType>(value.getType())) {
    return nullptr;
  }

  auto ptrType = mlir::LLVM::LLVMPointerType::get(builder.getContext());
  mlir::Value ptr = memrefToPointer(builder, loc, value, ptrType);

  if (!ptr) {
    return nullptr;
  }

  return pointerToBool(builder, loc, ptr);
}

auto toBool(mlir::OpBuilder &builder, mlir::Location loc, mlir::Value value)
    -> mlir::Value {
  if (!value) {
    return nullptr;
  }

  mlir::Type type = value.getType();

  if (type.isInteger(1)) {
    return value;
  }

  if (isIntegerLikeType(type)) {
    return integerToBool(builder, loc, value);
  }

  if (isFloatType(type)) {
    return floatToBool(builder, loc, value);
  }

  if (mlir::isa<mlir::LLVM::LLVMPointerType>(type)) {
    return pointerToBool(builder, loc, value);
  }

  if (mlir::isa<mlir::MemRefType>(type)) {
    return memrefToBool(builder, loc, value);
  }

  return nullptr;
}

auto toValue(mlir::OpBuilder &builder, mlir::Location loc, mlir::Value value,
             mlir::Type targetType, bool isSigned) -> mlir::Value {
  if (!value || !targetType) {
    return nullptr;
  }

  mlir::Type srcType = value.getType();

  if (srcType == targetType) {
    return value;
  }

  if (auto memref = mlir::dyn_cast<mlir::MemRefType>(targetType)) {
    return coerceMemRef(builder, loc, value, memref);
  }

  if (targetType.isInteger(1)) {
    return toBool(builder, loc, value);
  }

  if (targetType.isIndex()) {
    return toIndex(builder, loc, value, isSigned);
  }

  if (auto targetInt = mlir::dyn_cast<mlir::IntegerType>(targetType)) {
    return toInteger(builder, loc, value, targetInt, isSigned);
  }

  if (auto targetFloat = mlir::dyn_cast<mlir::FloatType>(targetType)) {
    return toFloat(builder, loc, value, targetFloat, isSigned);
  }

  if (mlir::isa<mlir::LLVM::LLVMPointerType>(targetType)) {
    return toPointer(builder, loc, value, targetType);
  }

  return nullptr;
}

auto scalarToBitcast(mlir::OpBuilder &builder, mlir::Location loc,
                     mlir::Value value, mlir::Type targetType) -> mlir::Value {
  if (!value || !targetType) {
    return nullptr;
  }

  mlir::Type srcType = value.getType();

  if (srcType == targetType) {
    return value;
  }

  if (srcType.isIndex() || targetType.isIndex()) {
    return nullptr;
  }

  unsigned srcWidth = 0;
  unsigned targetWidth = 0;

  if (auto srcInt = mlir::dyn_cast<mlir::IntegerType>(srcType)) {
    srcWidth = srcInt.getWidth();
  } else if (auto srcFloat = mlir::dyn_cast<mlir::FloatType>(srcType)) {
    srcWidth = srcFloat.getWidth();
  } else {
    return nullptr;
  }

  if (auto targetInt = mlir::dyn_cast<mlir::IntegerType>(targetType)) {
    targetWidth = targetInt.getWidth();
  } else if (auto targetFloat = mlir::dyn_cast<mlir::FloatType>(targetType)) {
    targetWidth = targetFloat.getWidth();
  } else {
    return nullptr;
  }

  if (srcWidth == 0 || srcWidth != targetWidth) {
    return nullptr;
  }

  return mlir::arith::BitcastOp::create(builder, loc, targetType, value)
      .getResult();
}

auto toBitcastValue(mlir::OpBuilder &builder, mlir::Location loc,
                    mlir::Value value, mlir::Type targetType, bool isSigned)
    -> mlir::Value {
  if (!value || !targetType) {
    return nullptr;
  }

  mlir::Type srcType = value.getType();

  if (srcType == targetType) {
    return value;
  }

  if (mlir::isa<mlir::LLVM::LLVMPointerType>(targetType)) {
    return toPointer(builder, loc, value, targetType);
  }

  if ((mlir::isa<mlir::LLVM::LLVMPointerType>(srcType) ||
       mlir::isa<mlir::MemRefType>(srcType)) &&
      (targetType.isIndex() || mlir::isa<mlir::IntegerType>(targetType))) {
    return toValue(builder, loc, value, targetType, isSigned);
  }

  if (mlir::Value bitcasted =
          scalarToBitcast(builder, loc, value, targetType)) {
    return bitcasted;
  }

  return toValue(builder, loc, value, targetType, isSigned);
}

} // namespace clang_mlir
