#include "./Types.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinTypes.h"

namespace cmlirc {

mlir::Type convertType(mlir::OpBuilder &builder, clang::QualType type) {
  const clang::Type *typePtr = type.getTypePtr();

  if (auto *builtinType = llvm::dyn_cast<clang::BuiltinType>(typePtr)) {
    return convertBuiltinType(builder, builtinType);
  }

  if (auto *pointerType = llvm::dyn_cast<clang::PointerType>(typePtr)) {
    return convertPointerType(builder, pointerType);
  }

  if (auto *arrayType = llvm::dyn_cast<clang::ArrayType>(typePtr)) {
    return convertArrayType(builder, arrayType);
  }

  llvm::errs() << "Unsupported type conversion for type: " << type.getAsString()
               << "\n";
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
    return nullptr;
  }
}

mlir::Type convertPointerType(mlir::OpBuilder &builder,
                              const clang::PointerType *type) {
  clang::QualType pointeeType = type->getPointeeType();
  mlir::Type elementType = convertType(builder, pointeeType);

  return mlir::UnrankedMemRefType::get(elementType, 0);
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

} // namespace cmlirc
