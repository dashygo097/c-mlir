#ifndef CMLIRC_CASTS_H
#define CMLIRC_CASTS_H

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/Builders.h"

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

inline mlir::Value toIndex(mlir::OpBuilder &builder, mlir::Location loc,
                           mlir::Value value) {
  if (value.getType().isIndex())
    return value;
  return mlir::arith::IndexCastOp::create(builder, loc, builder.getIndexType(),
                                          value)
      .getResult();
}

inline std::optional<int64_t> getConstantInt(mlir::Value value) {
  while (auto cast = value.getDefiningOp<mlir::arith::IndexCastOp>())
    value = cast.getIn();
  if (auto c = value.getDefiningOp<mlir::arith::ConstantOp>())
    if (auto ia = mlir::dyn_cast<mlir::IntegerAttr>(c.getValue()))
      return ia.getInt();
  return std::nullopt;
}

} // namespace cmlirc::detail

#endif // CMLIRC_CASTS_H
