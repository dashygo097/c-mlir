#ifndef CMLIRC_NUMERICS_H
#define CMLIRC_NUMERICS_H

#include "./Constants.h"
#include "./Operators.h"

namespace cmlirc::detail {

// Arithmetic
inline mlir::Value addi(mlir::OpBuilder &builder, mlir::Location loc,
                        mlir::Value value, int64_t amount) {
  mlir::Value amountValue = intConst(builder, loc, value.getType(), amount);
  return emitIntOp<mlir::arith::AddIOp>(builder, loc, value, amountValue);
}

inline mlir::Value addf(mlir::OpBuilder &builder, mlir::Location loc,
                        mlir::Value value, double amount) {
  mlir::Value amountValue = floatConst(builder, loc, value.getType(), amount);
  return emitFloatOp<mlir::arith::AddFOp>(builder, loc, value, amountValue);
}

inline mlir::Value subi(mlir::OpBuilder &builder, mlir::Location loc,
                        mlir::Value value, int64_t amount) {
  mlir::Value amountValue = intConst(builder, loc, value.getType(), amount);
  return emitIntOp<mlir::arith::SubIOp>(builder, loc, value, amountValue);
}

inline mlir::Value subf(mlir::OpBuilder &builder, mlir::Location loc,
                        mlir::Value value, double amount) {
  mlir::Value amountValue = floatConst(builder, loc, value.getType(), amount);
  return emitFloatOp<mlir::arith::SubFOp>(builder, loc, value, amountValue);
}

inline mlir::Value negi(mlir::OpBuilder &builder, mlir::Location loc,
                        mlir::Value value) {
  mlir::Value zero = intConst(builder, loc, value.getType(), 0);
  return emitIntOp<mlir::arith::SubIOp>(builder, loc, zero, value);
}

inline mlir::Value negf(mlir::OpBuilder &builder, mlir::Location loc,
                        mlir::Value value) {
  return emitFloatOp<mlir::arith::NegFOp>(builder, loc, value);
}

// Bitwise
inline mlir::Value noti(mlir::OpBuilder &builder, mlir::Location loc,
                        mlir::Value value) {
  mlir::Value allOnes = intConst(builder, loc, value.getType(), -1);
  return emitIntOp<mlir::arith::XOrIOp>(builder, loc, value, allOnes);
}

// Logical
inline mlir::Value notl(mlir::OpBuilder &builder, mlir::Location loc,
                        mlir::Value value) {
  return emitIntOp<mlir::arith::XOrIOp>(builder, loc, value,
                                        boolConst(builder, loc, true));
}

inline mlir::Value andl(mlir::OpBuilder &builder, mlir::Location loc,
                        mlir::Value lhs, mlir::Value rhs) {
  return emitIntOp<mlir::arith::AndIOp>(builder, loc, lhs, rhs);
}

inline mlir::Value orl(mlir::OpBuilder &builder, mlir::Location loc,
                       mlir::Value lhs, mlir::Value rhs) {
  return emitIntOp<mlir::arith::OrIOp>(builder, loc, lhs, rhs);
}

inline mlir::Value xorl(mlir::OpBuilder &builder, mlir::Location loc,
                        mlir::Value lhs, mlir::Value rhs) {
  return emitIntOp<mlir::arith::XOrIOp>(builder, loc, lhs, rhs);
}

} // namespace cmlirc::detail

#endif // CMLIRC_NUMERICS_H
