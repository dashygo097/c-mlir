#ifndef CHWC_UTILS_CONSTANT_H
#define CHWC_UTILS_CONSTANT_H

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Operation.h"
#include "llvm/Support/WithColor.h"

namespace chwc::utils {

inline auto intConst(mlir::OpBuilder &builder, mlir::Location loc,
                     mlir::Type type, int64_t value) -> mlir::Value {
  auto intType = mlir::dyn_cast<mlir::IntegerType>(type);
  if (!intType) {
    llvm::WithColor::error()
        << "chwc: hw.constant requires integer result type\n";
    return nullptr;
  }

  mlir::OperationState state(loc, "hw.constant");
  state.addAttribute("value", mlir::IntegerAttr::get(intType, value));
  state.addTypes(intType);

  mlir::Operation *op = builder.create(state);
  return op->getResult(0);
}

inline auto boolConst(mlir::OpBuilder &builder, mlir::Location loc, bool value)
    -> mlir::Value {
  return intConst(builder, loc, builder.getI1Type(), value ? 1 : 0);
}

inline auto zeroValue(mlir::OpBuilder &builder, mlir::Location loc,
                      mlir::Type type) -> mlir::Value {
  auto intType = mlir::dyn_cast<mlir::IntegerType>(type);
  if (intType) {
    return intConst(builder, loc, intType, 0);
  }

  llvm::WithColor::error()
      << "chwc: zeroValue only supports integer type here\n";
  return nullptr;
}

} // namespace chwc::utils

#endif // CHWC_UTILS_CONSTANT_H
