#ifndef CHWC_UTILS_CAST_H
#define CHWC_UTILS_CAST_H

#include "./Comb.h"
#include "./Constant.h"
#include "circt/Dialect/HW/HWTypes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"
#include "llvm/Support/WithColor.h"

namespace chwc::utils {

inline auto isBuiltinInteger(mlir::Type type) -> bool {
  return mlir::isa<mlir::IntegerType>(type);
}

inline auto isParameterizedInteger(mlir::Type type) -> bool {
  return mlir::isa<circt::hw::IntType>(type);
}

inline auto extractLowBits(mlir::OpBuilder &builder, mlir::Location loc,
                           mlir::Value value, mlir::Type targetType)
    -> mlir::Value {
  if (!value || !targetType) {
    return nullptr;
  }

  auto srcType = mlir::dyn_cast<mlir::IntegerType>(value.getType());
  auto dstType = mlir::dyn_cast<mlir::IntegerType>(targetType);

  if (!srcType || !dstType) {
    llvm::WithColor::error()
        << "chwc: comb.extract requires builtin integer types\n";
    return nullptr;
  }

  mlir::OperationState state(loc, "comb.extract");
  state.addOperands(value);
  state.addAttribute("lowBit", builder.getI32IntegerAttr(0));
  state.addTypes(dstType);

  mlir::Operation *op = builder.create(state);
  return op->getResult(0);
}

inline auto zeroExtendBuiltinInteger(mlir::OpBuilder &builder,
                                     mlir::Location loc, mlir::Value value,
                                     mlir::Type targetType) -> mlir::Value {
  if (!value || !targetType) {
    return nullptr;
  }

  auto srcType = mlir::dyn_cast<mlir::IntegerType>(value.getType());
  auto dstType = mlir::dyn_cast<mlir::IntegerType>(targetType);

  if (!srcType || !dstType) {
    llvm::WithColor::error()
        << "chwc: builtin integer cast requires builtin integer types\n";
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

  if (isParameterizedInteger(value.getType()) ||
      isParameterizedInteger(targetType)) {
    llvm::WithColor::error()
        << "chwc: cannot cast to/from parameterized hw integer; "
           "emit the value directly with the target type\n";
    return nullptr;
  }

  if (isBuiltinInteger(value.getType()) && isBuiltinInteger(targetType)) {
    return zeroExtendBuiltinInteger(builder, loc, value, targetType);
  }

  llvm::WithColor::error() << "chwc: unsupported hardware cast\n";
  return nullptr;
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

  if (auto intType = mlir::dyn_cast<mlir::IntegerType>(value.getType())) {
    if (intType.getWidth() == 1) {
      return value;
    }

    mlir::Value zero = intConst(builder, loc, intType, 0);
    if (!zero) {
      return nullptr;
    }

    return icmpNe(builder, loc, value, zero);
  }

  if (mlir::isa<circt::hw::IntType>(value.getType())) {
    mlir::Value zero = intConst(builder, loc, value.getType(), 0);
    if (!zero) {
      return nullptr;
    }

    return icmpNe(builder, loc, value, zero);
  }

  llvm::WithColor::error()
      << "chwc: boolean conversion expects integer-like value\n";
  return nullptr;
}

} // namespace chwc::utils

#endif // CHWC_UTILS_CAST_H
