#ifndef CMLIRC_NUMERICS_H
#define CMLIRC_NUMERICS_H

#include "./Constants.h"
#include "./Operators.h"

namespace cmlirc::detail {

inline mlir::Value addInt(mlir::OpBuilder &builder, mlir::Location loc,
                          mlir::Value value, int64_t amount) {
  mlir::Value amountValue = intConst(builder, loc, value.getType(), amount);
  return emitIntOp<mlir::arith::AddIOp>(builder, loc, value, amountValue);
}

inline mlir::Value subInt(mlir::OpBuilder &builder, mlir::Location loc,
                          mlir::Value value, int64_t amount) {
  mlir::Value amountValue = intConst(builder, loc, value.getType(), amount);
  return emitIntOp<mlir::arith::SubIOp>(builder, loc, value, amountValue);
}

inline mlir::Value addFloat(mlir::OpBuilder &builder, mlir::Location loc,
                            mlir::Value value, double amount) {
  mlir::Value amountValue = floatConst(builder, loc, value.getType(), amount);
  return emitIntOp<mlir::arith::AddFOp>(builder, loc, value, amountValue);
}

inline mlir::Value subFloat(mlir::OpBuilder &builder, mlir::Location loc,
                            mlir::Value value, double amount) {
  mlir::Value amountValue = floatConst(builder, loc, value.getType(), amount);
  return emitIntOp<mlir::arith::SubFOp>(builder, loc, value, amountValue);
}

} // namespace cmlirc::detail

#endif // CMLIRC_NUMERICS_H
