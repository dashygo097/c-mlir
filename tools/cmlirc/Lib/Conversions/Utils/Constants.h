#ifndef CMLIRC_CONSTANTS_H
#define CMLIRC_CONSTANTS_H

#include "./Types.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/Value.h"

namespace cmlirc::utils {

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
                     mlir::Type type, int64_t value) -> mlir::Value {
  if (type.isIndex()) {
    return indexConst(builder, loc, value);
  }

  if (!mlir::isa<mlir::IntegerType>(type)) {
    return nullptr;
  }

  return mlir::arith::ConstantOp::create(builder, loc, type,
                                         builder.getIntegerAttr(type, value))
      .getResult();
}

inline auto floatConst(mlir::OpBuilder &builder, mlir::Location loc,
                       mlir::Type type, double value) -> mlir::Value {
  if (!mlir::isa<mlir::FloatType>(type)) {
    return nullptr;
  }

  return mlir::arith::ConstantOp::create(builder, loc, type,
                                         builder.getFloatAttr(type, value))
      .getResult();
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

  return intConst(builder, loc, type, -1);
}

} // namespace cmlirc::utils

#endif // CMLIRC_CONSTANTS_H
