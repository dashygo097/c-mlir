#ifndef CHWC_UTILS_CASTS_H
#define CHWC_UTILS_CASTS_H

#include "./Constants.h"
#include "circt/Dialect/Comb/CombOps.h"
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

} // namespace chwc::utils

#endif // CHWC_UTILS_CASTS_H
