#ifndef CHWC_UTILS_ARRAY_H
#define CHWC_UTILS_ARRAY_H

#include "./Cast.h"
#include "./Constant.h"
#include "circt/Dialect/HW/HWTypes.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Support/WithColor.h"
#include <algorithm>

namespace chwc::utils {

inline auto getArrayIndexWidth(uint64_t size) -> unsigned {
  return std::max<unsigned>(1, llvm::Log2_64_Ceil(size));
}

inline auto castArrayIndex(mlir::OpBuilder &builder, mlir::Location loc,
                           mlir::Value index, uint64_t arraySize)
    -> mlir::Value {
  if (!index) {
    return nullptr;
  }

  return promoteValue(builder, loc, index,
                      builder.getIntegerType(getArrayIndexWidth(arraySize)));
}

inline auto createArray(mlir::OpBuilder &builder, mlir::Location loc,
                        mlir::Type arrayType,
                        llvm::ArrayRef<mlir::Value> values) -> mlir::Value {
  auto hwArrayType = mlir::dyn_cast<circt::hw::ArrayType>(arrayType);
  if (!hwArrayType) {
    llvm::WithColor::error() << "chwc: expected hw.array type\n";
    return nullptr;
  }

  if (values.size() != hwArrayType.getNumElements()) {
    llvm::WithColor::error() << "chwc: hw.array_create size mismatch\n";
    return nullptr;
  }

  mlir::OperationState state(loc, "hw.array_create");
  state.addOperands(values);
  state.addTypes(arrayType);

  mlir::Operation *op = builder.create(state);
  return op->getResult(0);
}

inline auto zeroArray(mlir::OpBuilder &builder, mlir::Location loc,
                      mlir::Type arrayType) -> mlir::Value {
  auto hwArrayType = mlir::dyn_cast<circt::hw::ArrayType>(arrayType);
  if (!hwArrayType) {
    llvm::WithColor::error() << "chwc: zeroArray expects hw.array type\n";
    return nullptr;
  }

  llvm::SmallVector<mlir::Value, 16> values;

  for (uint64_t i = 0; i < hwArrayType.getNumElements(); ++i) {
    values.push_back(zeroValue(builder, loc, hwArrayType.getElementType()));
  }

  return createArray(builder, loc, arrayType, values);
}

inline auto arrayGet(mlir::OpBuilder &builder, mlir::Location loc,
                     mlir::Value arrayValue, mlir::Value index) -> mlir::Value {
  if (!arrayValue || !index) {
    return nullptr;
  }

  auto arrayType = mlir::dyn_cast<circt::hw::ArrayType>(arrayValue.getType());
  if (!arrayType) {
    llvm::WithColor::error() << "chwc: hw.array_get input must be hw.array\n";
    return nullptr;
  }

  index = castArrayIndex(builder, loc, index, arrayType.getNumElements());
  if (!index) {
    return nullptr;
  }

  mlir::OperationState state(loc, "hw.array_get");
  state.addOperands({arrayValue, index});
  state.addTypes(arrayType.getElementType());

  mlir::Operation *op = builder.create(state);
  return op->getResult(0);
}

inline auto arrayInject(mlir::OpBuilder &builder, mlir::Location loc,
                        mlir::Value arrayValue, mlir::Value index,
                        mlir::Value element) -> mlir::Value {
  if (!arrayValue || !index || !element) {
    return nullptr;
  }

  auto arrayType = mlir::dyn_cast<circt::hw::ArrayType>(arrayValue.getType());
  if (!arrayType) {
    llvm::WithColor::error()
        << "chwc: hw.array_inject input must be hw.array\n";
    return nullptr;
  }

  index = castArrayIndex(builder, loc, index, arrayType.getNumElements());
  if (!index) {
    return nullptr;
  }

  element = promoteValue(builder, loc, element, arrayType.getElementType());
  if (!element) {
    return nullptr;
  }

  mlir::OperationState state(loc, "hw.array_inject");
  state.addOperands({arrayValue, index, element});
  state.addTypes(arrayValue.getType());

  mlir::Operation *op = builder.create(state);
  return op->getResult(0);
}

} // namespace chwc::utils

#endif // CHWC_UTILS_ARRAY_H
