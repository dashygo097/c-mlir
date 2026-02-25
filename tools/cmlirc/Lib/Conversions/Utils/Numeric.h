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

inline mlir::Value addOne(mlir::OpBuilder &builder, mlir::Location loc,
                          mlir::Value value) {
  return addInt(builder, loc, value, 1);
}

} // namespace cmlirc::detail

#endif // CMLIRC_NUMERIC_H
