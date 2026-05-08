#ifndef CHWC_UTILS_CAST_H
#define CHWC_UTILS_CAST_H

#include "./Comb.h"
#include "./Constant.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"
#include "llvm/Support/WithColor.h"

namespace chwc::utils {

inline auto extractLowBits(mlir::OpBuilder &builder, mlir::Location loc,
                           mlir::Value value, mlir::Type targetType)
    -> mlir::Value {
  if (!value || !targetType) {
    return nullptr;
  }

  auto srcType = mlir::dyn_cast<mlir::IntegerType>(value.getType());
  auto dstType = mlir::dyn_cast<mlir::IntegerType>(targetType);

  if (!srcType || !dstType) {
    llvm::WithColor::error() << "chwc: comb.extract requires integer types\n";
    return nullptr;
  }

  mlir::OperationState state(loc, "comb.extract");
  state.addOperands(value);
  state.addAttribute("lowBit", builder.getI32IntegerAttr(0));
  state.addTypes(dstType);

  mlir::Operation *op = builder.create(state);
  return op->getResult(0);
}

inline auto zeroExtend(mlir::OpBuilder &builder, mlir::Location loc,
                       mlir::Value value, mlir::Type targetType)
    -> mlir::Value {
  if (!value || !targetType) {
    return nullptr;
  }

  auto srcType = mlir::dyn_cast<mlir::IntegerType>(value.getType());
  auto dstType = mlir::dyn_cast<mlir::IntegerType>(targetType);

  if (!srcType || !dstType) {
    llvm::WithColor::error()
        << "chwc: comb zero-extension requires integer types\n";
    return nullptr;
  }

  unsigned srcWidth = srcType.getWidth();
  unsigned dstWidth = dstType.getWidth();

  if (srcWidth == dstWidth) {
    return value;
  }

  if (srcWidth > dstWidth) {
    return extractLowBits(builder, loc, value, dstType);
  }

  unsigned padWidth = dstWidth - srcWidth;
  mlir::Value zero =
      intConst(builder, loc, builder.getIntegerType(padWidth), 0);
  if (!zero) {
    return nullptr;
  }

  mlir::OperationState state(loc, "comb.concat");
  state.addOperands({zero, value});
  state.addTypes(dstType);

  mlir::Operation *op = builder.create(state);
  return op->getResult(0);
}

inline auto promoteValue(mlir::OpBuilder &builder, mlir::Location loc,
                         mlir::Value value, mlir::Type targetType)
    -> mlir::Value {
  if (!value) {
    return nullptr;
  }

  if (!targetType || value.getType() == targetType) {
    return value;
  }

  auto srcType = mlir::dyn_cast<mlir::IntegerType>(value.getType());
  auto dstType = mlir::dyn_cast<mlir::IntegerType>(targetType);

  if (!srcType || !dstType) {
    llvm::WithColor::error()
        << "chwc: only integer cast is supported in hardware path\n";
    return nullptr;
  }

  return zeroExtend(builder, loc, value, targetType);
}

inline auto truncateValue(mlir::OpBuilder &builder, mlir::Location loc,
                          mlir::Value value, mlir::Type targetType)
    -> mlir::Value {
  return promoteValue(builder, loc, value, targetType);
}

inline auto toBool(mlir::OpBuilder &builder, mlir::Location loc,
                   mlir::Value value) -> mlir::Value {
  if (!value) {
    return nullptr;
  }

  auto intType = mlir::dyn_cast<mlir::IntegerType>(value.getType());
  if (!intType) {
    llvm::WithColor::error()
        << "chwc: boolean conversion expects integer value\n";
    return nullptr;
  }

  if (intType.getWidth() == 1) {
    return value;
  }

  mlir::Value zero = intConst(builder, loc, intType, 0);
  if (!zero) {
    return nullptr;
  }

  return icmpNe(builder, loc, value, zero);
}

} // namespace chwc::utils

#endif // CHWC_UTILS_CAST_H
