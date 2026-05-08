#ifndef CHWC_UTILS_CONSTANT_H
#define CHWC_UTILS_CONSTANT_H

#include "circt/Dialect/HW/HWAttributes.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/HW/HWTypes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "llvm/Support/WithColor.h"

namespace chwc::utils {

inline auto isIntegerLikeType(mlir::Type type) -> bool {
  return mlir::isa<mlir::IntegerType>(type) ||
         mlir::isa<circt::hw::IntType>(type);
}

inline auto builtinIntConst(mlir::OpBuilder &builder, mlir::Location loc,
                            mlir::IntegerType type, int64_t value)
    -> mlir::Value {
  auto valueAttr = mlir::IntegerAttr::get(type, value);

  auto op = circt::hw::ConstantOp::create(builder, loc, type, valueAttr);
  return op.getResult();
}

inline auto paramIntConst(mlir::OpBuilder &builder, mlir::Location loc,
                          circt::hw::IntType type, int64_t value)
    -> mlir::Value {
  auto text = builder.getStringAttr(std::to_string(value));

  auto valueAttr =
      circt::hw::ParamVerbatimAttr::get(builder.getContext(), text, type);

  auto op = circt::hw::ParamValueOp::create(builder, loc, type, valueAttr);
  return op.getResult();
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
