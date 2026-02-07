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

  llvm::outs() << "Unsupported type: " << type.getAsString();
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
    return builder.getI32Type();
  }
}

mlir::Type convertArrayType(mlir::OpBuilder &builder,
                            const clang::ArrayType *type) {
  clang::QualType elementQualType = type->getElementType();
  mlir::Type elementType = convertType(builder, elementQualType);

  if (auto *constArrayType = llvm::dyn_cast<clang::ConstantArrayType>(type)) {
    int64_t size = constArrayType->getSize().getSExtValue();
    return mlir::MemRefType::get({size}, elementType);
  }

  return mlir::UnrankedMemRefType::get(elementType, 0);
}

mlir::Type convertPointerType(mlir::OpBuilder &builder,
                              const clang::PointerType *type) {
  clang::QualType pointeeQualType = type->getPointeeType();
  mlir::Type elementType = convertType(builder, pointeeQualType);

  return mlir::UnrankedMemRefType::get(elementType, 0);
}

} // namespace cmlirc
