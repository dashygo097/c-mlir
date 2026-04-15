#ifndef CMLIRC_OPERATORS_H
#define CMLIRC_OPERATORS_H

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"

namespace cmlirc::utils {

template <typename IntOp, typename FloatOp>
auto emitOp(mlir::OpBuilder &builder, mlir::Location loc, mlir::Value lhs,
            mlir::Value rhs) -> mlir::Value {
  mlir::Type ty = lhs.getType();
  if (mlir::isa<mlir::IntegerType>(ty)) {
    return IntOp::create(builder, loc, lhs, rhs).getResult();
  }
  if (mlir::isa<mlir::FloatType>(ty)) {
    return FloatOp::create(builder, loc, lhs, rhs).getResult();
  }
  return nullptr;
}

template <typename IntOp>
auto emitIntOp(mlir::OpBuilder &builder, mlir::Location loc, mlir::Value lhs,
               mlir::Value rhs) -> mlir::Value {
  if (mlir::isa<mlir::IntegerType>(lhs.getType())) {
    return IntOp::create(builder, loc, lhs, rhs).getResult();
  }
  return nullptr;
}

template <typename FloatOp>
auto emitFloatOp(mlir::OpBuilder &builder, mlir::Location loc,
                 mlir::Value value) -> mlir::Value {
  if (mlir::isa<mlir::FloatType>(value.getType())) {
    return FloatOp::create(builder, loc, value).getResult();
  }
  return nullptr;
}

template <typename FloatOp>
auto emitFloatOp(mlir::OpBuilder &builder, mlir::Location loc, mlir::Value lhs,
                 mlir::Value rhs) -> mlir::Value {
  if (mlir::isa<mlir::FloatType>(lhs.getType())) {
    return FloatOp::create(builder, loc, lhs, rhs).getResult();
  }
  return nullptr;
}

inline auto emitCmpOp(mlir::OpBuilder &builder, mlir::Location loc,
                      mlir::arith::CmpIPredicate iPred,
                      mlir::arith::CmpFPredicate fPred, mlir::Value lhs,
                      mlir::Value rhs) -> mlir::Value {
  mlir::Type ty = lhs.getType();
  if (mlir::isa<mlir::IntegerType>(ty)) {
    return mlir::arith::CmpIOp::create(builder, loc, iPred, lhs, rhs)
        .getResult();
  }
  if (mlir::isa<mlir::FloatType>(ty)) {
    return mlir::arith::CmpFOp::create(builder, loc, fPred, lhs, rhs)
        .getResult();
  }
  return nullptr;
}

} // namespace cmlirc::utils

#endif // CMLIRC_OPERATORS_H
