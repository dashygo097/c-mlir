#ifndef CMLIRC_CASTS_H
#define CMLIRC_CASTS_H

#include "./Constants.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Value.h"

namespace cmlirc::utils {

inline auto toInteger(mlir::OpBuilder &builder, mlir::Location loc,
                      mlir::Value value, mlir::IntegerType targetType,
                      bool isSigned = true) -> mlir::Value {
  if (!value) {
    return nullptr;
  }

  mlir::Type srcType = value.getType();

  if (srcType == targetType) {
    return value;
  }

  if (srcType.isIndex()) {
    return mlir::arith::IndexCastOp::create(builder, loc, targetType, value)
        .getResult();
  }

  if (auto srcInt = mlir::dyn_cast<mlir::IntegerType>(srcType)) {
    if (srcInt.getWidth() < targetType.getWidth()) {
      return isSigned
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
    return mlir::LLVM::PtrToIntOp::create(builder, loc, targetType, value)
        .getResult();
  }

  return nullptr;
}

inline auto toIndex(mlir::OpBuilder &builder, mlir::Location loc,
                    mlir::Value value, bool isSigned = true) -> mlir::Value {
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

  return mlir::arith::IndexCastOp::create(builder, loc, builder.getIndexType(),
                                          integer)
      .getResult();
}

inline auto toFloat(mlir::OpBuilder &builder, mlir::Location loc,
                    mlir::Value value, mlir::FloatType targetType,
                    bool isSigned = true) -> mlir::Value {
  if (!value) {
    return nullptr;
  }

  mlir::Type srcType = value.getType();

  if (srcType == targetType) {
    return value;
  }

  if (srcType.isIndex()) {
    value = mlir::arith::IndexCastOp::create(builder, loc, builder.getI64Type(),
                                             value)
                .getResult();
    srcType = value.getType();
  }

  if (auto srcInt = mlir::dyn_cast<mlir::IntegerType>(srcType)) {
    return isSigned
               ? mlir::arith::SIToFPOp::create(builder, loc, targetType, value)
                     .getResult()
               : mlir::arith::UIToFPOp::create(builder, loc, targetType, value)
                     .getResult();
  }

  if (auto srcFloat = mlir::dyn_cast<mlir::FloatType>(srcType)) {
    if (srcFloat.getWidth() < targetType.getWidth()) {
      return mlir::arith::ExtFOp::create(builder, loc, targetType, value)
          .getResult();
    }

    if (srcFloat.getWidth() > targetType.getWidth()) {
      return mlir::arith::TruncFOp::create(builder, loc, targetType, value)
          .getResult();
    }

    return value;
  }

  return nullptr;
}

inline auto toBool(mlir::OpBuilder &builder, mlir::Location loc,
                   mlir::Value value) -> mlir::Value {
  if (!value) {
    return nullptr;
  }

  mlir::Type type = value.getType();

  if (type.isInteger(1)) {
    return value;
  }

  if (isIntegerLikeType(type)) {
    return mlir::arith::CmpIOp::create(builder, loc,
                                       mlir::arith::CmpIPredicate::ne, value,
                                       zeroConst(builder, loc, type))
        .getResult();
  }

  if (isFloatType(type)) {
    return mlir::arith::CmpFOp::create(builder, loc,
                                       mlir::arith::CmpFPredicate::ONE, value,
                                       zeroConst(builder, loc, type))
        .getResult();
  }

  if (mlir::isa<mlir::LLVM::LLVMPointerType>(type)) {
    mlir::Value ptrAsInt = mlir::LLVM::PtrToIntOp::create(
                               builder, loc, builder.getI64Type(), value)
                               .getResult();

    return mlir::arith::CmpIOp::create(
               builder, loc, mlir::arith::CmpIPredicate::ne, ptrAsInt,
               intConst(builder, loc, builder.getI64Type(), 0))
        .getResult();
  }

  return nullptr;
}

inline auto castValue(mlir::OpBuilder &builder, mlir::Location loc,
                      mlir::Value value, mlir::Type targetType,
                      bool isSigned = true) -> mlir::Value {
  if (!value) {
    return nullptr;
  }

  mlir::Type srcType = value.getType();

  if (srcType == targetType) {
    return value;
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
    if (srcType.isIndex()) {
      value = mlir::arith::IndexCastOp::create(builder, loc,
                                               builder.getI64Type(), value)
                  .getResult();
      srcType = value.getType();
    }

    if (mlir::isa<mlir::IntegerType>(srcType)) {
      return mlir::LLVM::IntToPtrOp::create(builder, loc, targetType, value)
          .getResult();
    }
  }

  return nullptr;
}

} // namespace cmlirc::utils

#endif // CMLIRC_CASTS_H
