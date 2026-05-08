#ifndef CHWC_UTILS_CONSTANT_H
#define CHWC_UTILS_CONSTANT_H

#include "circt/Dialect/HW/HWTypes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Operation.h"
#include "llvm/Support/WithColor.h"

namespace chwc::utils {

inline auto isIntegerLikeType(mlir::Type type) -> bool {
  return mlir::isa<mlir::IntegerType>(type) ||
         mlir::isa<circt::hw::IntType>(type);
}

inline auto builtinIntConst(mlir::OpBuilder &builder, mlir::Location loc,
                            mlir::IntegerType type, int64_t value)
    -> mlir::Value {
  mlir::OperationState state(loc, "hw.constant");
  state.addAttribute("value", mlir::IntegerAttr::get(type, value));
  state.addTypes(type);

  mlir::Operation *op = builder.create(state);
  return op->getResult(0);
}

inline auto paramIntConst(mlir::OpBuilder &builder, mlir::Location loc,
                          circt::hw::IntType type, int64_t value)
    -> mlir::Value {
  mlir::OperationState state(loc, "hw.param.value");

  state.addAttribute("value",
                     mlir::IntegerAttr::get(builder.getIntegerType(64), value));
  state.addTypes(type);

  mlir::Operation *op = builder.create(state);
  return op->getResult(0);
}

inline auto intConst(mlir::OpBuilder &builder, mlir::Location loc,
                     mlir::Type type, int64_t value) -> mlir::Value {
  if (auto intType = mlir::dyn_cast<mlir::IntegerType>(type)) {
    return builtinIntConst(builder, loc, intType, value);
  }

  if (auto hwIntType = mlir::dyn_cast<circt::hw::IntType>(type)) {
    return paramIntConst(builder, loc, hwIntType, value);
  }

  llvm::WithColor::error()
      << "chwc: constant requires integer-like result type\n";
  return nullptr;
}

inline auto boolConst(mlir::OpBuilder &builder, mlir::Location loc, bool value)
    -> mlir::Value {
  return intConst(builder, loc, builder.getI1Type(), value ? 1 : 0);
}

inline auto zeroValue(mlir::OpBuilder &builder, mlir::Location loc,
                      mlir::Type type) -> mlir::Value {
  if (isIntegerLikeType(type)) {
    return intConst(builder, loc, type, 0);
  }

  llvm::WithColor::error()
      << "chwc: zeroValue only supports integer-like type\n";
  return nullptr;
}

} // namespace chwc::utils

#endif // CHWC_UTILS_CONSTANT_H
