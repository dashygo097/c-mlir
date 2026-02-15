#include "./Types.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinTypes.h"

namespace cmlirc {

mlir::Type convertType(mlir::OpBuilder &builder, clang::QualType type) {
  type = type.getCanonicalType();
  const clang::Type *typePtr = type.getTypePtr();

  if (auto *builtinType = llvm::dyn_cast<clang::BuiltinType>(typePtr)) {
    return convertBuiltinType(builder, builtinType);
  }
  if (auto *arrayType = llvm::dyn_cast<clang::ArrayType>(typePtr)) {
    return convertArrayType(builder, arrayType);
  }
  if (auto *pointerType = llvm::dyn_cast<clang::PointerType>(typePtr)) {
    return convertPointerType(builder, pointerType);
  }

  llvm::errs() << "Unsupported type: " << type.getAsString();
  return nullptr;
}

mlir::Type convertBuiltinType(mlir::OpBuilder &builder,
                              const clang::BuiltinType *type) {
  switch (type->getKind()) {
  case clang::BuiltinType::Void:
    return builder.getNoneType();

  case clang::BuiltinType::Bool:
    return builder.getI1Type();

  case clang::BuiltinType::Char_S:
  case clang::BuiltinType::Char_U:
  case clang::BuiltinType::SChar:
  case clang::BuiltinType::UChar:
    return builder.getI8Type();

  case clang::BuiltinType::Short:
  case clang::BuiltinType::UShort:
    return builder.getI16Type();

  case clang::BuiltinType::Int:
  case clang::BuiltinType::UInt:
    return builder.getI32Type();

  case clang::BuiltinType::Long:
  case clang::BuiltinType::ULong:
  case clang::BuiltinType::LongLong:
  case clang::BuiltinType::ULongLong:
    return builder.getI64Type();

  case clang::BuiltinType::Float:
    return builder.getF32Type();

  case clang::BuiltinType::Double:
    return builder.getF64Type();

  default:
    llvm::errs() << "Unsupported builtin type: " << type << "\n";
  }

  return nullptr;
}

mlir::Type convertArrayType(mlir::OpBuilder &builder,
                            const clang::ArrayType *type) {

  llvm::SmallVector<int64_t, 4> dimensions;
  clang::QualType currentType = clang::QualType(type, 0);

  while (auto *arrayType =
             llvm::dyn_cast<clang::ArrayType>(currentType.getTypePtr())) {
    if (auto *constArrayType =
            llvm::dyn_cast<clang::ConstantArrayType>(arrayType)) {
      int64_t size = constArrayType->getSize().getSExtValue();
      dimensions.push_back(size);
      currentType = constArrayType->getElementType();
    } else {
      dimensions.push_back(-1);
      currentType = arrayType->getElementType();
    }
  }

  mlir::Type elementType = convertType(builder, currentType);

  return mlir::MemRefType::get(dimensions, elementType);
}

mlir::Type convertPointerType(mlir::OpBuilder &builder,
                              const clang::PointerType *type) {
  clang::QualType pointeeType = type->getPointeeType();

  if (auto *_ = llvm::dyn_cast<clang::ArrayType>(pointeeType.getTypePtr())) {
    llvm::SmallVector<int64_t, 4> dimensions;
    clang::QualType currentType = pointeeType;

    while (auto *arrType =
               llvm::dyn_cast<clang::ArrayType>(currentType.getTypePtr())) {
      if (auto *constArrayType =
              llvm::dyn_cast<clang::ConstantArrayType>(arrType)) {
        int64_t size = constArrayType->getSize().getSExtValue();
        dimensions.push_back(size);
        currentType = constArrayType->getElementType();
      } else {
        dimensions.push_back(mlir::ShapedType::kDynamic);
        currentType = arrType->getElementType();
      }
    }

    mlir::Type elementType = convertType(builder, currentType);

    dimensions.insert(dimensions.begin(), mlir::ShapedType::kDynamic);

    return mlir::MemRefType::get(dimensions, elementType);
  }

  mlir::Type elementType = convertType(builder, pointeeType);
  return mlir::MemRefType::get({mlir::ShapedType::kDynamic}, elementType);
}

// helpers
mlir::Value convertToBool(mlir::OpBuilder &builder, mlir::Value value) {
  mlir::Location loc = builder.getUnknownLoc();
  mlir::Type type = value.getType();

  if (type.isInteger(1)) {
    return value;
  }

  if (auto intType = mlir::dyn_cast<mlir::IntegerType>(type)) {
    mlir::Value zero = mlir::arith::ConstantOp::create(
        builder, loc, type, builder.getIntegerAttr(type, 0));
    return mlir::arith::CmpIOp::create(
               builder, loc, mlir::arith::CmpIPredicate::ne, value, zero)
        .getResult();
  }

  if (auto floatType = mlir::dyn_cast<mlir::FloatType>(type)) {
    mlir::Value zero = mlir::arith::ConstantOp::create(
        builder, loc, type, builder.getFloatAttr(type, 0.0));
    return mlir::arith::CmpFOp::create(
               builder, loc,
               mlir::arith::CmpFPredicate::ONE, // ONE = Ordered Not Equal
               value, zero)
        .getResult();
  }

  llvm::errs() << "Cannot convert type to bool\n";
  return nullptr;
}

} // namespace cmlirc
