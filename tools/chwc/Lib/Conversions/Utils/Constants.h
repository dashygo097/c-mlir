#ifndef CHWC_UTILS_CONSTANTS_H
#define CHWC_UTILS_CONSTANTS_H

#include "circt/Dialect/HW/HWOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "llvm/Support/WithColor.h"

namespace chwc::utils {

inline auto intConst(mlir::OpBuilder &builder, mlir::Location loc,
                     mlir::Type type, uint64_t value) -> mlir::Value {
  auto intType = mlir::dyn_cast<mlir::IntegerType>(type);
  if (!intType) {
    llvm::WithColor::error()
        << "chwc: hw.constant requires integer result type\n";
    return nullptr;
  }

  return circt::hw::ConstantOp::create(builder, loc, type, value).getResult();
}

inline auto zeroValue(mlir::OpBuilder &builder, mlir::Location loc,
                      mlir::Type type) -> mlir::Value {
  return intConst(builder, loc, type, 0);
}

inline auto oneValue(mlir::OpBuilder &builder, mlir::Location loc,
                     mlir::Type type) -> mlir::Value {
  return intConst(builder, loc, type, 1);
}

} // namespace chwc::utils

#endif // CHWC_UTILS_CONSTANTS_H
