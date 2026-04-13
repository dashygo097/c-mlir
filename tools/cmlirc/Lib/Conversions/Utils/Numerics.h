#ifndef CMLIRC_NUMERICS_H
#define CMLIRC_NUMERICS_H

#include "./Constants.h"
#include "./Operators.h"

namespace cmlirc::detail {
// Arithmetic
inline auto addi(mlir::OpBuilder &builder, mlir::Location loc,
                 mlir::Value value, int64_t amount) -> mlir::Value {
  mlir::Value amountValue = intConst(builder, loc, value.getType(), amount);
  return emitIntOp<mlir::arith::AddIOp>(builder, loc, value, amountValue);
}

inline auto addf(mlir::OpBuilder &builder, mlir::Location loc,
                 mlir::Value value, double amount) -> mlir::Value {
  mlir::Value amountValue = floatConst(builder, loc, value.getType(), amount);
  return emitFloatOp<mlir::arith::AddFOp>(builder, loc, value, amountValue);
}

inline auto subi(mlir::OpBuilder &builder, mlir::Location loc,
                 mlir::Value value, int64_t amount) -> mlir::Value {
  mlir::Value amountValue = intConst(builder, loc, value.getType(), amount);
  return emitIntOp<mlir::arith::SubIOp>(builder, loc, value, amountValue);
}

inline auto subf(mlir::OpBuilder &builder, mlir::Location loc,
                 mlir::Value value, double amount) -> mlir::Value {
  mlir::Value amountValue = floatConst(builder, loc, value.getType(), amount);
  return emitFloatOp<mlir::arith::SubFOp>(builder, loc, value, amountValue);
}

inline auto sub(mlir::OpBuilder &builder, mlir::Location loc, mlir::Value lhs,
                mlir::Value rhs) -> mlir::Value {
  if (mlir::isa<mlir::IntegerType>(lhs.getType())) {
    return emitIntOp<mlir::arith::SubIOp>(builder, loc, lhs, rhs);
  }
  if (mlir::isa<mlir::FloatType>(lhs.getType())) {
    return emitFloatOp<mlir::arith::SubFOp>(builder, loc, lhs, rhs);
  }
  return nullptr;
}

inline auto negi(mlir::OpBuilder &builder, mlir::Location loc,
                 mlir::Value value) -> mlir::Value {
  mlir::Value zero = intConst(builder, loc, value.getType(), 0);
  return emitIntOp<mlir::arith::SubIOp>(builder, loc, zero, value);
}

inline auto negf(mlir::OpBuilder &builder, mlir::Location loc,
                 mlir::Value value) -> mlir::Value {
  return emitFloatOp<mlir::arith::NegFOp>(builder, loc, value);
}

inline auto neg(mlir::OpBuilder &builder, mlir::Location loc, mlir::Value value)
    -> mlir::Value {
  if (mlir::isa<mlir::IntegerType>(value.getType())) {
    return negi(builder, loc, value);
  }
  if (mlir::isa<mlir::FloatType>(value.getType())) {
    return negf(builder, loc, value);
  }
  return nullptr;
}

// Bitwise
inline auto noti(mlir::OpBuilder &builder, mlir::Location loc,
                 mlir::Value value) -> mlir::Value {
  mlir::Value allOnes = intConst(builder, loc, value.getType(), -1);
  return emitIntOp<mlir::arith::XOrIOp>(builder, loc, value, allOnes);
}

inline auto andi(mlir::OpBuilder &builder, mlir::Location loc, mlir::Value lhs,
                 mlir::Value rhs) -> mlir::Value {
  return emitIntOp<mlir::arith::AndIOp>(builder, loc, lhs, rhs);
}

inline auto ori(mlir::OpBuilder &builder, mlir::Location loc, mlir::Value lhs,
                mlir::Value rhs) -> mlir::Value {
  return emitIntOp<mlir::arith::OrIOp>(builder, loc, lhs, rhs);
}

inline auto xori(mlir::OpBuilder &builder, mlir::Location loc, mlir::Value lhs,
                 mlir::Value rhs) -> mlir::Value {
  return emitIntOp<mlir::arith::XOrIOp>(builder, loc, lhs, rhs);
}

} // namespace cmlirc::detail

#endif // CMLIRC_NUMERICS_H
