#ifndef CHWC_UTILS_CONSTANT_H
#define CHWC_UTILS_CONSTANT_H

#include "circt/Dialect/HW/HWOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "llvm/ADT/APInt.h"
#include "llvm/Support/WithColor.h"
#include <cstdint>
#include <limits>

namespace chwc::utils {

inline auto normalizeUIntToWidth(mlir::Type type, uint64_t value) -> uint64_t {
  auto intType = mlir::dyn_cast<mlir::IntegerType>(type);
  if (!intType) {
    return value;
  }

  unsigned width = intType.getWidth();
  if (width >= 64) {
    return value;
  }

  if (width == 0) {
    return 0;
  }

  return value & ((uint64_t{1} << width) - 1);
}

inline auto intConst(mlir::OpBuilder &builder, mlir::Location loc,
                     mlir::Type type, uint64_t value) -> mlir::Value {
  auto intType = mlir::dyn_cast<mlir::IntegerType>(type);
  if (!intType) {
    llvm::WithColor::error()
        << "chwc: hw.constant requires integer result type\n";
    return nullptr;
  }

  value = normalizeUIntToWidth(type, value);

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

inline auto signedIntConst(mlir::OpBuilder &builder, mlir::Location loc,
                           mlir::Type type, int64_t value) -> mlir::Value {
  return intConst(builder, loc, type, static_cast<uint64_t>(value));
}

inline auto apIntConst(mlir::OpBuilder &builder, mlir::Location loc,
                       mlir::Type type, llvm::APInt value) -> mlir::Value {
  auto intType = mlir::dyn_cast<mlir::IntegerType>(type);
  if (!intType) {
    llvm::WithColor::error()
        << "chwc: hw.constant requires integer result type\n";
    return nullptr;
  }

  value = value.zextOrTrunc(intType.getWidth());
  return intConst(builder, loc, type, value.getZExtValue());
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

#endif // CHWC_UTILS_CONSTANT_H
