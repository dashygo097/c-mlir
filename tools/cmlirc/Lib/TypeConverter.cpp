#include "./TypeConverter.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinTypes.h"

namespace cmlirc {

TypeConverter::TypeConverter(mlir::OpBuilder &builder) : builder_(builder) {}

mlir::Type TypeConverter::convertType(clang::QualType type) {
  const clang::Type *typePtr = type.getTypePtr();

  if (auto *builtinType = llvm::dyn_cast<clang::BuiltinType>(typePtr)) {
    return convertBuiltinType(builtinType);
  }

  if (auto *pointerType = llvm::dyn_cast<clang::PointerType>(typePtr)) {
    return convertPointerType(pointerType);
  }

  if (auto *arrayType = llvm::dyn_cast<clang::ArrayType>(typePtr)) {
    return convertArrayType(arrayType);
  }

  return builder_.getI32Type();
}

mlir::Type TypeConverter::convertBuiltinType(const clang::BuiltinType *type) {
  switch (type->getKind()) {
  case clang::BuiltinType::Void:
    return builder_.getNoneType();

  case clang::BuiltinType::Bool:
    return builder_.getI1Type();

  case clang::BuiltinType::Char_S:
  case clang::BuiltinType::Char_U:
  case clang::BuiltinType::SChar:
  case clang::BuiltinType::UChar:
    return builder_.getI8Type();

  case clang::BuiltinType::Short:
  case clang::BuiltinType::UShort:
    return builder_.getI16Type();

  case clang::BuiltinType::Int:
  case clang::BuiltinType::UInt:
    return builder_.getI32Type();

  case clang::BuiltinType::Long:
  case clang::BuiltinType::ULong:
  case clang::BuiltinType::LongLong:
  case clang::BuiltinType::ULongLong:
    return builder_.getI64Type();

  case clang::BuiltinType::Float:
    return builder_.getF32Type();

  case clang::BuiltinType::Double:
    return builder_.getF64Type();

  default:
    return builder_.getI32Type();
  }
}

mlir::Type TypeConverter::convertPointerType(const clang::PointerType *type) {
  clang::QualType pointeeType = type->getPointeeType();
  mlir::Type elementType = convertType(pointeeType);

  return mlir::UnrankedMemRefType::get(elementType, 0);
}

mlir::Type TypeConverter::convertArrayType(const clang::ArrayType *type) {
  clang::QualType elementQualType = type->getElementType();
  mlir::Type elementType = convertType(elementQualType);

  if (auto *constArrayType = llvm::dyn_cast<clang::ConstantArrayType>(type)) {
    int64_t size = constArrayType->getSize().getSExtValue();
    return mlir::MemRefType::get({size}, elementType);
  }

  return mlir::UnrankedMemRefType::get(elementType, 0);
}

} // namespace cmlirc
