#ifndef CMLIRC_CONSTANTS_H
#define CMLIRC_CONSTANTS_H

#include "mlir/Dialect/Arith/IR/Arith.h"

namespace cmlirc::detail {
inline mlir::Value indexConst(mlir::OpBuilder &builder, mlir::Location loc,
                              int64_t value) {
  return mlir::arith::ConstantOp::create(builder, loc, builder.getIndexType(),
                                         builder.getIndexAttr(value))
      .getResult();
}

inline mlir::Value boolConst(mlir::OpBuilder &builder, mlir::Location loc,
                             bool value) {
  return mlir::arith::ConstantOp::create(builder, loc, builder.getI1Type(),
                                         builder.getBoolAttr(value))
      .getResult();
}

inline mlir::Value intConst(mlir::OpBuilder &builder, mlir::Location loc,
                            mlir::Type type, int64_t value) {
  return mlir::arith::ConstantOp::create(builder, loc, type,
                                         builder.getIntegerAttr(type, value))
      .getResult();
}

inline mlir::Value floatConst(mlir::OpBuilder &builder, mlir::Location loc,
                              mlir::Type type, double value) {
  return mlir::arith::ConstantOp::create(builder, loc, type,
                                         builder.getFloatAttr(type, value))
      .getResult();
}
} // namespace cmlirc::detail

#endif // CMLIRC_CONSTANTS_H
