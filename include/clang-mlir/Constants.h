#ifndef CLANG_MLIR_CONSTANTS_H
#define CLANG_MLIR_CONSTANTS_H

#include "Types.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/Value.h"
#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/APSInt.h"

namespace clang_mlir {

inline auto indexConst(mlir::OpBuilder &builder, mlir::Location loc,
                       int64_t value) -> mlir::Value {
  return mlir::arith::ConstantOp::create(builder, loc, builder.getIndexType(),
                                         builder.getIndexAttr(value))
      .getResult();
}

inline auto boolConst(mlir::OpBuilder &builder, mlir::Location loc, bool value)
    -> mlir::Value {
  return mlir::arith::ConstantOp::create(builder, loc, builder.getI1Type(),
                                         builder.getBoolAttr(value))
      .getResult();
}

inline auto intConst(mlir::OpBuilder &builder, mlir::Location loc,
                     mlir::Type type, const llvm::APInt &value) -> mlir::Value {
  if (!type || (!type.isIndex() && !mlir::isa<mlir::IntegerType>(type))) {
    return nullptr;
  }
  unsigned width = type.isIndex() ? 64 : type.getIntOrFloatBitWidth();
  if (value.getBitWidth() != width) {
    return nullptr;
  }
  return mlir::arith::ConstantOp::create(builder, loc, type,
                                         builder.getIntegerAttr(type, value))
      .getResult();
}

inline auto intConst(mlir::OpBuilder &builder, mlir::Location loc,
                     mlir::Type type, int64_t value) -> mlir::Value {
  if (!type || (!type.isIndex() && !mlir::isa<mlir::IntegerType>(type))) {
    return nullptr;
  }
  unsigned width = type.isIndex() ? 64 : type.getIntOrFloatBitWidth();
  return intConst(builder, loc, type, llvm::APInt(width, value, true));
}

inline auto intConst(mlir::OpBuilder &builder, mlir::Location loc,
                     mlir::Type type, const llvm::APSInt &value)
    -> mlir::Value {
  auto intType = mlir::dyn_cast_or_null<mlir::IntegerType>(type);
  if (!intType) {
    return nullptr;
  }
  auto bits = value.isSigned() ? value.sextOrTrunc(intType.getWidth())
                               : value.zextOrTrunc(intType.getWidth());
  return intConst(builder, loc, type, bits);
}

inline auto floatConst(mlir::OpBuilder &builder, mlir::Location loc,
                       mlir::Type type, const llvm::APFloat &value)
    -> mlir::Value {
  auto floating = mlir::dyn_cast_or_null<mlir::FloatType>(type);
  if (!floating || &floating.getFloatSemantics() != &value.getSemantics()) {
    return nullptr;
  }

  return mlir::arith::ConstantOp::create(builder, loc, type,
                                         builder.getFloatAttr(type, value))
      .getResult();
}

inline auto floatConst(mlir::OpBuilder &builder, mlir::Location loc,
                       mlir::Type type, double value) -> mlir::Value {
  auto floating = mlir::dyn_cast_or_null<mlir::FloatType>(type);
  if (!floating) {
    return nullptr;
  }
  llvm::APFloat number(value);
  bool losesInfo;
  number.convert(floating.getFloatSemantics(),
                 llvm::APFloat::rmNearestTiesToEven, &losesInfo);
  return floatConst(builder, loc, type, number);
}

inline auto numericConst(mlir::OpBuilder &builder, mlir::Location loc,
                         mlir::Type type, int64_t intValue, double floatValue)
    -> mlir::Value {
  if (isIntegerLikeType(type)) {
    return intConst(builder, loc, type, intValue);
  }

  if (isFloatType(type)) {
    return floatConst(builder, loc, type, floatValue);
  }

  return nullptr;
}

inline auto zeroConst(mlir::OpBuilder &builder, mlir::Location loc,
                      mlir::Type type) -> mlir::Value {
  return numericConst(builder, loc, type, 0, 0.0);
}

inline auto oneConst(mlir::OpBuilder &builder, mlir::Location loc,
                     mlir::Type type) -> mlir::Value {
  return numericConst(builder, loc, type, 1, 1.0);
}

inline auto allOnesConst(mlir::OpBuilder &builder, mlir::Location loc,
                         mlir::Type type) -> mlir::Value {
  if (!isIntegerLikeType(type)) {
    return nullptr;
  }

  unsigned width = type.isIndex() ? 64 : type.getIntOrFloatBitWidth();
  return intConst(builder, loc, type, llvm::APInt::getAllOnes(width));
}

} // namespace clang_mlir

#endif // CLANG_MLIR_CONSTANTS_H
