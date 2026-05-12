#ifndef CMLIRC_NUMERICS_H
#define CMLIRC_NUMERICS_H

#include "./Constants.h"
#include "./Operators.h"

namespace cmlirc::utils {

inline auto add(mlir::OpBuilder &builder, mlir::Location loc, mlir::Value lhs,
                mlir::Value rhs) -> mlir::Value {
  return emitNumericOp<mlir::arith::AddIOp, mlir::arith::AddFOp>(builder, loc,
                                                                 lhs, rhs);
}

inline auto sub(mlir::OpBuilder &builder, mlir::Location loc, mlir::Value lhs,
                mlir::Value rhs) -> mlir::Value {
  return emitNumericOp<mlir::arith::SubIOp, mlir::arith::SubFOp>(builder, loc,
                                                                 lhs, rhs);
}

inline auto mul(mlir::OpBuilder &builder, mlir::Location loc, mlir::Value lhs,
                mlir::Value rhs) -> mlir::Value {
  return emitNumericOp<mlir::arith::MulIOp, mlir::arith::MulFOp>(builder, loc,
                                                                 lhs, rhs);
}

inline auto divs(mlir::OpBuilder &builder, mlir::Location loc, mlir::Value lhs,
                 mlir::Value rhs) -> mlir::Value {
  return emitNumericOp<mlir::arith::DivSIOp, mlir::arith::DivFOp>(builder, loc,
                                                                  lhs, rhs);
}

inline auto rems(mlir::OpBuilder &builder, mlir::Location loc, mlir::Value lhs,
                 mlir::Value rhs) -> mlir::Value {
  return emitIntOp<mlir::arith::RemSIOp>(builder, loc, lhs, rhs);
}

inline auto neg(mlir::OpBuilder &builder, mlir::Location loc, mlir::Value value)
    -> mlir::Value {
  if (!value) {
    return nullptr;
  }

  mlir::Type type = value.getType();

  if (isIntegerLikeType(type)) {
    mlir::Value zero = zeroConst(builder, loc, type);
    return emitIntOp<mlir::arith::SubIOp>(builder, loc, zero, value);
  }

  if (isFloatType(type)) {
    return emitFloatOp<mlir::arith::NegFOp>(builder, loc, value);
  }

  return nullptr;
}

inline auto bitNot(mlir::OpBuilder &builder, mlir::Location loc,
                   mlir::Value value) -> mlir::Value {
  if (!value || !isIntegerLikeType(value.getType())) {
    return nullptr;
  }

  mlir::Value allOnes = allOnesConst(builder, loc, value.getType());
  return emitIntOp<mlir::arith::XOrIOp>(builder, loc, value, allOnes);
}

inline auto bitAnd(mlir::OpBuilder &builder, mlir::Location loc,
                   mlir::Value lhs, mlir::Value rhs) -> mlir::Value {
  return emitIntOp<mlir::arith::AndIOp>(builder, loc, lhs, rhs);
}

inline auto bitOr(mlir::OpBuilder &builder, mlir::Location loc, mlir::Value lhs,
                  mlir::Value rhs) -> mlir::Value {
  return emitIntOp<mlir::arith::OrIOp>(builder, loc, lhs, rhs);
}

inline auto bitXor(mlir::OpBuilder &builder, mlir::Location loc,
                   mlir::Value lhs, mlir::Value rhs) -> mlir::Value {
  return emitIntOp<mlir::arith::XOrIOp>(builder, loc, lhs, rhs);
}

inline auto shl(mlir::OpBuilder &builder, mlir::Location loc, mlir::Value lhs,
                mlir::Value rhs) -> mlir::Value {
  return emitIntOp<mlir::arith::ShLIOp>(builder, loc, lhs, rhs);
}

inline auto shrs(mlir::OpBuilder &builder, mlir::Location loc, mlir::Value lhs,
                 mlir::Value rhs) -> mlir::Value {
  return emitIntOp<mlir::arith::ShRSIOp>(builder, loc, lhs, rhs);
}

inline auto addi(mlir::OpBuilder &builder, mlir::Location loc,
                 mlir::Value value, int64_t amount) -> mlir::Value {
  if (!value || !isIntegerLikeType(value.getType())) {
    return nullptr;
  }

  return add(builder, loc, value,
             intConst(builder, loc, value.getType(), amount));
}

inline auto addf(mlir::OpBuilder &builder, mlir::Location loc,
                 mlir::Value value, double amount) -> mlir::Value {
  if (!value || !isFloatType(value.getType())) {
    return nullptr;
  }

  return add(builder, loc, value,
             floatConst(builder, loc, value.getType(), amount));
}

inline auto subi(mlir::OpBuilder &builder, mlir::Location loc,
                 mlir::Value value, int64_t amount) -> mlir::Value {
  if (!value || !isIntegerLikeType(value.getType())) {
    return nullptr;
  }

  return sub(builder, loc, value,
             intConst(builder, loc, value.getType(), amount));
}

inline auto subf(mlir::OpBuilder &builder, mlir::Location loc,
                 mlir::Value value, double amount) -> mlir::Value {
  if (!value || !isFloatType(value.getType())) {
    return nullptr;
  }

  return sub(builder, loc, value,
             floatConst(builder, loc, value.getType(), amount));
}

inline auto inc(mlir::OpBuilder &builder, mlir::Location loc, mlir::Value value)
    -> mlir::Value {
  if (!value || !isIntegerLikeType(value.getType())) {
    return nullptr;
  }

  return addi(builder, loc, value, 1);
}

inline auto dec(mlir::OpBuilder &builder, mlir::Location loc, mlir::Value value)
    -> mlir::Value {
  if (!value || !isIntegerLikeType(value.getType())) {
    return nullptr;
  }

  return subi(builder, loc, value, 1);
}

} // namespace cmlirc::utils

#endif // CMLIRC_NUMERICS_H
