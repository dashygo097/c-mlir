#ifndef CMLIRC_OPERATORS_H
#define CMLIRC_OPERATORS_H

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"

namespace cmlirc::detail {

template <typename IntOp, typename FloatOp>
mlir::Value emitOp(mlir::OpBuilder &b, mlir::Location loc, mlir::Value lhs,
                   mlir::Value rhs) {
  mlir::Type ty = lhs.getType();
  if (mlir::isa<mlir::IntegerType>(ty))
    return IntOp::create(b, loc, lhs, rhs).getResult();
  if (mlir::isa<mlir::FloatType>(ty))
    return FloatOp::create(b, loc, lhs, rhs).getResult();
  return nullptr;
}

template <typename IntOp>
mlir::Value emitIntOp(mlir::OpBuilder &b, mlir::Location loc, mlir::Value lhs,
                      mlir::Value rhs) {
  if (mlir::isa<mlir::IntegerType>(lhs.getType()))
    return IntOp::create(b, loc, lhs, rhs).getResult();
  return nullptr;
}

template <typename FloatOp>
mlir::Value emitFloatOp(mlir::OpBuilder &b, mlir::Location loc, mlir::Value lhs,
                        mlir::Value rhs) {
  if (mlir::isa<mlir::FloatType>(lhs.getType()))
    return FloatOp::create(b, loc, lhs, rhs).getResult();
  return nullptr;
}

inline mlir::Value emitCmpOp(mlir::OpBuilder &b, mlir::Location loc,
                             mlir::arith::CmpIPredicate iPred,
                             mlir::arith::CmpFPredicate fPred, mlir::Value lhs,
                             mlir::Value rhs) {
  mlir::Type ty = lhs.getType();
  if (mlir::isa<mlir::IntegerType>(ty))
    return mlir::arith::CmpIOp::create(b, loc, iPred, lhs, rhs).getResult();
  if (mlir::isa<mlir::FloatType>(ty))
    return mlir::arith::CmpFOp::create(b, loc, fPred, lhs, rhs).getResult();
  return nullptr;
}

} // namespace cmlirc::detail

#endif // CMLIRC_OPERATORS_H
