#ifndef CHWC_UTILS_CASTS_H
#define CHWC_UTILS_CASTS_H

#include "./Constant.h"
#include "circt/Dialect/Comb/CombOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/BuiltinTypes.h"
#include "llvm/Support/WithColor.h"

namespace chwc::utils {

inline auto toBool(mlir::OpBuilder &builder, mlir::Location loc,
                   mlir::Value value) -> mlir::Value {
  if (!value) {
    return nullptr;
  }

  auto intType = mlir::dyn_cast<mlir::IntegerType>(value.getType());
  if (!intType) {
    llvm::WithColor::error()
        << "chwc: only integer value can be converted to bool\n";
    return nullptr;
  }

  if (intType.getWidth() == 1) {
    return value;
  }

  mlir::Value zero = zeroValue(builder, loc, value.getType());
  if (!zero) {
    return nullptr;
  }

  return circt::comb::ICmpOp::create(
             builder, loc, circt::comb::ICmpPredicate::ne, value, zero)
      .getResult();
}

inline auto promoteValue(mlir::OpBuilder &builder, mlir::Location loc,
                         mlir::Value value, mlir::Type targetType)
    -> mlir::Value {
  if (!value || !targetType) {
    return nullptr;
  }

  if (value.getType() == targetType) {
    return value;
  }

  auto srcType = mlir::dyn_cast<mlir::IntegerType>(value.getType());
  auto dstType = mlir::dyn_cast<mlir::IntegerType>(targetType);

  if (!srcType || !dstType) {
    llvm::WithColor::error() << "chwc: only integer promotion is supported\n";
    return value;
  }

  if (srcType.getWidth() < dstType.getWidth()) {
    return mlir::arith::ExtUIOp::create(builder, loc, targetType, value)
        .getResult();
  }

  if (srcType.getWidth() > dstType.getWidth()) {
    return mlir::arith::TruncIOp::create(builder, loc, targetType, value)
        .getResult();
  }

  return value;
}

inline auto truncateValue(mlir::OpBuilder &builder, mlir::Location loc,
                          mlir::Value value, mlir::Type targetType)
    -> mlir::Value {
  return promoteValue(builder, loc, value, targetType);
}

} // namespace chwc::utils

#endif // CHWC_UTILS_CASTS_H
