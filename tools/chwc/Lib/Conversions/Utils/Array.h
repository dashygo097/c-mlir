#ifndef CHWC_UTILS_ARRAY_H
#define CHWC_UTILS_ARRAY_H

#include "../../Converter.h"
#include "./Cast.h"
#include "./Constant.h"
#include "circt/Dialect/HW/HWTypes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Operation.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Support/WithColor.h"

namespace chwc::utils {

inline auto getArrayIndexWidth(uint64_t size) -> unsigned {
  if (size <= 1) {
    return 1;
  }

  return llvm::Log2_64_Ceil(size);
}

inline auto getArrayIndexType(mlir::OpBuilder &builder, uint64_t size)
    -> mlir::IntegerType {
  return builder.getIntegerType(getArrayIndexWidth(size));
}

inline auto coerceArrayIndex(mlir::OpBuilder &builder, mlir::Location loc,
                             mlir::Value index, uint64_t size) -> mlir::Value {
  if (!index) {
    return nullptr;
  }

  return promoteValue(builder, loc, index, getArrayIndexType(builder, size));
}

inline auto createArray(mlir::OpBuilder &builder, mlir::Location loc,
                        llvm::ArrayRef<mlir::Value> elements,
                        mlir::Type arrayType) -> mlir::Value {
  if (elements.empty()) {
    llvm::WithColor::error() << "chwc: cannot create empty hw.array\n";
    return nullptr;
  }

  llvm::SmallVector<mlir::Value, 16> operands;
  for (auto it = elements.rbegin(); it != elements.rend(); ++it) {
    operands.push_back(*it);
  }

  mlir::OperationState opState(loc, "hw.array_create");
  opState.addOperands(operands);
  opState.addTypes(arrayType);

  mlir::Operation *op = builder.create(opState);
  if (!op || op->getNumResults() == 0) {
    llvm::WithColor::error() << "chwc: failed to create hw.array_create\n";
    return nullptr;
  }

  return op->getResult(0);
}

inline auto zeroAnyValue(mlir::OpBuilder &builder, mlir::Location loc,
                         mlir::Type type) -> mlir::Value;

inline auto zeroArrayValue(mlir::OpBuilder &builder, mlir::Location loc,
                           mlir::Type type) -> mlir::Value {
  auto arrayType = mlir::dyn_cast<circt::hw::ArrayType>(type);
  if (!arrayType) {
    llvm::WithColor::error() << "chwc: zeroArrayValue requires hw.array type\n";
    return nullptr;
  }

  llvm::SmallVector<mlir::Value, 16> elements;
  for (uint64_t i = 0, e = arrayType.getNumElements(); i < e; ++i) {
    mlir::Value element =
        zeroAnyValue(builder, loc, arrayType.getElementType());
    if (!element) {
      return nullptr;
    }

    elements.push_back(element);
  }

  return createArray(builder, loc, elements, type);
}

inline auto zeroAnyValue(mlir::OpBuilder &builder, mlir::Location loc,
                         mlir::Type type) -> mlir::Value {
  if (mlir::isa<circt::hw::ArrayType>(type)) {
    return zeroArrayValue(builder, loc, type);
  }

  return zeroValue(builder, loc, type);
}

inline auto zeroFieldValue(mlir::OpBuilder &builder, mlir::Location loc,
                           const HWFieldInfo &fieldInfo) -> mlir::Value {
  return zeroAnyValue(builder, loc, fieldInfo.type);
}

inline auto arrayGet(mlir::OpBuilder &builder, mlir::Location loc,
                     mlir::Value arrayValue, mlir::Value index,
                     mlir::Type elementType) -> mlir::Value {
  if (!arrayValue || !index || !elementType) {
    return nullptr;
  }

  mlir::OperationState opState(loc, "hw.array_get");
  opState.addOperands({arrayValue, index});
  opState.addTypes(elementType);

  mlir::Operation *op = builder.create(opState);
  if (!op || op->getNumResults() == 0) {
    llvm::WithColor::error() << "chwc: failed to create hw.array_get\n";
    return nullptr;
  }

  return op->getResult(0);
}

inline auto arrayInject(mlir::OpBuilder &builder, mlir::Location loc,
                        mlir::Value arrayValue, mlir::Value index,
                        mlir::Value elementValue, mlir::Type arrayType)
    -> mlir::Value {
  if (!arrayValue || !index || !elementValue || !arrayType) {
    return nullptr;
  }

  mlir::OperationState opState(loc, "hw.array_inject");
  opState.addOperands({arrayValue, index, elementValue});
  opState.addTypes(arrayType);

  mlir::Operation *op = builder.create(opState);
  if (!op || op->getNumResults() == 0) {
    llvm::WithColor::error() << "chwc: failed to create hw.array_inject\n";
    return nullptr;
  }

  return op->getResult(0);
}

} // namespace chwc::utils

#endif // CHWC_UTILS_ARRAY_H
