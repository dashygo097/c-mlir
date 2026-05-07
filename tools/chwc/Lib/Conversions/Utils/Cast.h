#ifndef CHWC_UTILS_CAST_H
#define CHWC_UTILS_CAST_H

#include "./Constant.h"
#include "circt/Dialect/Comb/CombOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Operation.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/WithColor.h"

namespace chwc::utils {

inline auto createCombExtract(mlir::OpBuilder &builder, mlir::Location loc,
                              mlir::Value value, mlir::Type targetType,
                              uint32_t lowBit = 0) -> mlir::Value {
  if (!value || !targetType) {
    return nullptr;
  }

  auto srcType = mlir::dyn_cast<mlir::IntegerType>(value.getType());
  auto dstType = mlir::dyn_cast<mlir::IntegerType>(targetType);

  if (!srcType || !dstType) {
    llvm::WithColor::error() << "chwc: comb.extract requires integer types\n";
    return value;
  }

  if (dstType.getWidth() > srcType.getWidth()) {
    llvm::WithColor::error()
        << "chwc: comb.extract result cannot be wider than input\n";
    return value;
  }

  if (lowBit + dstType.getWidth() > srcType.getWidth()) {
    llvm::WithColor::error()
        << "chwc: comb.extract range exceeds input width\n";
    return value;
  }

  if (srcType.getWidth() == dstType.getWidth() && lowBit == 0) {
    return value;
  }

  mlir::OperationState opState(loc, "comb.extract");
  opState.addOperands(value);
  opState.addTypes(targetType);
  opState.addAttribute("lowBit", builder.getI32IntegerAttr(lowBit));

  mlir::Operation *op = builder.create(opState);
  if (!op || op->getNumResults() == 0) {
    llvm::WithColor::error() << "chwc: failed to create comb.extract\n";
    return value;
  }

  return op->getResult(0);
}

inline auto createCombConcat(mlir::OpBuilder &builder, mlir::Location loc,
                             llvm::ArrayRef<mlir::Value> operands,
                             mlir::Type targetType) -> mlir::Value {
  if (operands.empty() || !targetType) {
    return nullptr;
  }

  mlir::OperationState opState(loc, "comb.concat");
  opState.addOperands(operands);
  opState.addTypes(targetType);

  mlir::Operation *op = builder.create(opState);
  if (!op || op->getNumResults() == 0) {
    llvm::WithColor::error() << "chwc: failed to create comb.concat\n";
    return nullptr;
  }

  return op->getResult(0);
}

inline auto zeroExtendValue(mlir::OpBuilder &builder, mlir::Location loc,
                            mlir::Value value, mlir::Type targetType)
    -> mlir::Value {
  if (!value || !targetType) {
    return nullptr;
  }

  auto srcType = mlir::dyn_cast<mlir::IntegerType>(value.getType());
  auto dstType = mlir::dyn_cast<mlir::IntegerType>(targetType);

  if (!srcType || !dstType) {
    llvm::WithColor::error() << "chwc: zero extension requires integer types\n";
    return value;
  }

  uint32_t srcWidth = srcType.getWidth();
  uint32_t dstWidth = dstType.getWidth();

  if (srcWidth == dstWidth) {
    return value;
  }

  if (srcWidth > dstWidth) {
    return createCombExtract(builder, loc, value, targetType, 0);
  }

  mlir::Type padType = builder.getIntegerType(dstWidth - srcWidth);
  mlir::Value pad = zeroValue(builder, loc, padType);
  if (!pad) {
    return nullptr;
  }

  llvm::SmallVector<mlir::Value, 2> operands;
  operands.push_back(pad);
  operands.push_back(value);

  return createCombConcat(builder, loc, operands, targetType);
}

inline auto truncateValue(mlir::OpBuilder &builder, mlir::Location loc,
                          mlir::Value value, mlir::Type targetType)
    -> mlir::Value {
  if (!value || !targetType) {
    return nullptr;
  }

  auto srcType = mlir::dyn_cast<mlir::IntegerType>(value.getType());
  auto dstType = mlir::dyn_cast<mlir::IntegerType>(targetType);

  if (!srcType || !dstType) {
    llvm::WithColor::error() << "chwc: truncation requires integer types\n";
    return value;
  }

  if (srcType.getWidth() <= dstType.getWidth()) {
    return zeroExtendValue(builder, loc, value, targetType);
  }

  return createCombExtract(builder, loc, value, targetType, 0);
}

inline auto promoteValue(mlir::OpBuilder &builder, mlir::Location loc,
                         mlir::Value value, mlir::Type targetType)
    -> mlir::Value {
  if (!value || !targetType) {
    return nullptr;
  }

  if (value.getType() == targetType) {
    return value;
  }

  auto srcType = mlir::dyn_cast<mlir::IntegerType>(value.getType());
  auto dstType = mlir::dyn_cast<mlir::IntegerType>(targetType);

  if (!srcType || !dstType) {
    llvm::WithColor::error() << "chwc: only integer promotion is supported\n";
    return value;
  }

  if (srcType.getWidth() < dstType.getWidth()) {
    return zeroExtendValue(builder, loc, value, targetType);
  }

  if (srcType.getWidth() > dstType.getWidth()) {
    return truncateValue(builder, loc, value, targetType);
  }

  return value;
}

inline auto toBool(mlir::OpBuilder &builder, mlir::Location loc,
                   mlir::Value value) -> mlir::Value {
  if (!value) {
    return nullptr;
  }

  auto intType = mlir::dyn_cast<mlir::IntegerType>(value.getType());
  if (!intType) {
    llvm::WithColor::error()
        << "chwc: only integer value can be converted to bool\n";
    return nullptr;
  }

  if (intType.getWidth() == 1) {
    return value;
  }

  mlir::Value zero = zeroValue(builder, loc, value.getType());
  if (!zero) {
    return nullptr;
  }

  return circt::comb::ICmpOp::create(
             builder, loc, circt::comb::ICmpPredicate::ne, value, zero)
      .getResult();
}

} // namespace chwc::utils

#endif // CHWC_UTILS_CAST_H
