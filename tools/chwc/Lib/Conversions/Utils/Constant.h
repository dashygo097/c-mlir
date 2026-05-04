#ifndef CHWC_UTILS_CONSTANTS_H
#define CHWC_UTILS_CONSTANTS_H

#include "circt/Dialect/HW/HWOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "llvm/Support/WithColor.h"
#include <cstdint>
#include <limits>

namespace chwc::utils {

inline auto intConst(mlir::OpBuilder &builder, mlir::Location loc,
                     mlir::Type type, uint64_t value) -> mlir::Value {
  auto intType = mlir::dyn_cast<mlir::IntegerType>(type);
  if (!intType) {
    llvm::WithColor::error()
        << "chwc: hw.constant requires integer result type\n";
    return nullptr;
  }

  mlir::Block *block = builder.getInsertionBlock();
  if (block) {
    for (mlir::Operation &op : *block) {
      auto constantOp = mlir::dyn_cast<circt::hw::ConstantOp>(&op);
      if (!constantOp) {
        continue;
      }

      if (constantOp.getType() != type) {
        continue;
      }

      if (constantOp.getValue().getZExtValue() != value) {
        continue;
      }

      return constantOp.getResult();
    }
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

inline auto allOnesValue(mlir::OpBuilder &builder, mlir::Location loc,
                         mlir::Type type) -> mlir::Value {
  auto intType = mlir::dyn_cast<mlir::IntegerType>(type);
  if (!intType) {
    llvm::WithColor::error()
        << "chwc: all-ones constant requires integer type\n";
    return nullptr;
  }

  unsigned width = intType.getWidth();
  uint64_t value = 0;

  if (width >= 64) {
    value = std::numeric_limits<uint64_t>::max();
  } else {
    value = (uint64_t{1} << width) - 1;
  }

  return intConst(builder, loc, type, value);
}

} // namespace chwc::utils

#endif // CHWC_UTILS_CONSTANTS_H
