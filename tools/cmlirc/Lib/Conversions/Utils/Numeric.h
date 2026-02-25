#ifndef CMLIRC_NUMERIC_H
#define CMLIRC_NUMERIC_H

#include "./Casts.h"

namespace cmlirc::detail {

inline mlir::Value addOne(mlir::OpBuilder &builder, mlir::Location loc,
                          mlir::Value value) {
  mlir::Value one = intConst(builder, loc, value.getType(), 1);
  return mlir::arith::AddIOp::create(builder, loc, value, one).getResult();
}

} // namespace cmlirc::detail

#endif // CMLIRC_NUMERIC_H
