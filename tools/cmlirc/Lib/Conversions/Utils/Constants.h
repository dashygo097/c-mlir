#ifndef CMLIRC_CONSTANTS_H
#define CMLIRC_CONSTANTS_H

#include "mlir/Dialect/Arith/IR/Arith.h"

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
  return mlir::arith::ConstantOp::create(builder, loc, type,
                                         builder.getIntegerAttr(type, value))
      .getResult();
}

inline auto floatConst(mlir::OpBuilder &builder, mlir::Location loc,
                       mlir::Type type, double value) -> mlir::Value {
  return mlir::arith::ConstantOp::create(builder, loc, type,
                                         builder.getFloatAttr(type, value))
      .getResult();
}
} // namespace cmlirc::utils

#endif // CMLIRC_CONSTANTS_H
