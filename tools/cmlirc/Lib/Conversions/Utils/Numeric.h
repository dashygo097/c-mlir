#ifndef CMLIRC_NUMERIC_H
#define CMLIRC_NUMERIC_H

#include "./Casts.h"

namespace cmlirc::detail {

inline mlir::Value addInt(mlir::OpBuilder &builder, mlir::Location loc,
                          mlir::Value value, int64_t amount) {
  mlir::Value amountVal = intConst(builder, loc, value.getType(), amount);
  return mlir::arith::AddIOp::create(builder, loc, value, amountVal)
      .getResult();
}

inline mlir::Value subInt(mlir::OpBuilder &builder, mlir::Location loc,
                          mlir::Value value, int64_t amount) {
  mlir::Value amountVal = intConst(builder, loc, value.getType(), amount);
  return mlir::arith::SubIOp::create(builder, loc, value, amountVal)
      .getResult();
}

inline mlir::Value mulInt(mlir::OpBuilder &builder, mlir::Location loc,
                          mlir::Value value, int64_t amount) {
  mlir::Value amountVal = intConst(builder, loc, value.getType(), amount);
  return mlir::arith::MulIOp::create(builder, loc, value, amountVal)
      .getResult();
}

inline mlir::Value divInt(mlir::OpBuilder &builder, mlir::Location loc,
                          mlir::Value value, int64_t amount) {
  mlir::Value amountVal = intConst(builder, loc, value.getType(), amount);
  return mlir::arith::DivSIOp::create(builder, loc, value, amountVal)
      .getResult();
}

inline mlir::Value addFloat(mlir::OpBuilder &builder, mlir::Location loc,
                            mlir::Value value, double amount) {
  mlir::Value amountVal = floatConst(builder, loc, value.getType(), amount);
  return mlir::arith::AddFOp::create(builder, loc, value, amountVal)
      .getResult();
}

inline mlir::Value subFloat(mlir::OpBuilder &builder, mlir::Location loc,
                            mlir::Value value, double amount) {
  mlir::Value amountVal = floatConst(builder, loc, value.getType(), amount);
  return mlir::arith::SubFOp::create(builder, loc, value, amountVal)
      .getResult();
}

inline mlir::Value mulFloat(mlir::OpBuilder &builder, mlir::Location loc,
                            mlir::Value value, double amount) {
  mlir::Value amountVal = floatConst(builder, loc, value.getType(), amount);
  return mlir::arith::MulFOp::create(builder, loc, value, amountVal)
      .getResult();
}

inline mlir::Value divFloat(mlir::OpBuilder &builder, mlir::Location loc,
                            mlir::Value value, double amount) {
  mlir::Value amountVal = floatConst(builder, loc, value.getType(), amount);
  return mlir::arith::DivFOp::create(builder, loc, value, amountVal)
      .getResult();
}

} // namespace cmlirc::detail

#endif // CMLIRC_NUMERIC_H
