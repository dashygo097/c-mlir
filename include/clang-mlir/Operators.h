#ifndef CLANG_MLIR_OPERATORS_H
#define CLANG_MLIR_OPERATORS_H

#include "Types.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"

namespace clang_mlir {

inline auto sameType(mlir::Value lhs, mlir::Value rhs) -> bool {
  return lhs && rhs && lhs.getType() == rhs.getType();
}

template <typename Op>
inline auto emitUnaryOp(mlir::OpBuilder &builder, mlir::Location loc,
                        mlir::Value value) -> mlir::Value {
  if (!value) {
    return nullptr;
  }

  return Op::create(builder, loc, value).getResult();
}

template <typename Op>
inline auto emitBinaryOp(mlir::OpBuilder &builder, mlir::Location loc,
                         mlir::Value lhs, mlir::Value rhs) -> mlir::Value {
  if (!sameType(lhs, rhs)) {
    return nullptr;
  }

  return Op::create(builder, loc, lhs, rhs).getResult();
}

template <typename IntOp>
inline auto emitIntOp(mlir::OpBuilder &builder, mlir::Location loc,
                      mlir::Value lhs, mlir::Value rhs) -> mlir::Value {
  if (!sameType(lhs, rhs) || !isIntegerLikeType(lhs.getType())) {
    return nullptr;
  }

  return IntOp::create(builder, loc, lhs, rhs).getResult();
}

template <typename FloatOp>
inline auto
emitFloatOp(mlir::OpBuilder &builder, mlir::Location loc, mlir::Value value,
            mlir::arith::FastMathFlags flags = mlir::arith::FastMathFlags::none)
    -> mlir::Value {
  if (!value || !isFloatType(value.getType())) {
    return nullptr;
  }

  return FloatOp::create(builder, loc, value, flags).getResult();
}

template <typename FloatOp>
inline auto
emitFloatOp(mlir::OpBuilder &builder, mlir::Location loc, mlir::Value lhs,
            mlir::Value rhs,
            mlir::arith::FastMathFlags flags = mlir::arith::FastMathFlags::none)
    -> mlir::Value {
  if (!sameType(lhs, rhs) || !isFloatType(lhs.getType())) {
    return nullptr;
  }

  return FloatOp::create(builder, loc, lhs, rhs, flags).getResult();
}

template <typename IntOp, typename FloatOp>
inline auto emitNumericOp(mlir::OpBuilder &builder, mlir::Location loc,
                          mlir::Value lhs, mlir::Value rhs,
                          mlir::arith::FastMathFlags flags =
                              mlir::arith::FastMathFlags::none) -> mlir::Value {
  if (!sameType(lhs, rhs)) {
    return nullptr;
  }

  mlir::Type type = lhs.getType();

  if (isIntegerLikeType(type)) {
    return IntOp::create(builder, loc, lhs, rhs).getResult();
  }

  if (isFloatType(type)) {
    return FloatOp::create(builder, loc, lhs, rhs, flags).getResult();
  }

  return nullptr;
}

template <typename IntOp, typename FloatOp>
inline auto emitOp(mlir::OpBuilder &builder, mlir::Location loc,
                   mlir::Value lhs, mlir::Value rhs) -> mlir::Value {
  return emitNumericOp<IntOp, FloatOp>(builder, loc, lhs, rhs);
}

inline auto
emitCmpOp(mlir::OpBuilder &builder, mlir::Location loc,
          mlir::arith::CmpIPredicate iPred, mlir::arith::CmpFPredicate fPred,
          mlir::Value lhs, mlir::Value rhs,
          mlir::arith::FastMathFlags flags = mlir::arith::FastMathFlags::none)
    -> mlir::Value {
  if (!sameType(lhs, rhs)) {
    return nullptr;
  }

  mlir::Type type = lhs.getType();

  if (isIntegerLikeType(type)) {
    return mlir::arith::CmpIOp::create(builder, loc, iPred, lhs, rhs)
        .getResult();
  }

  if (isFloatType(type)) {
    return mlir::arith::CmpFOp::create(builder, loc, fPred, lhs, rhs, flags)
        .getResult();
  }

  return nullptr;
}

} // namespace clang_mlir

#endif // CLANG_MLIR_OPERATORS_H
