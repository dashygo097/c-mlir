#include "../../Converter.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"

namespace cmlirc {

mlir::Type CMLIRConverter::convertType(clang::QualType type) {
  type = type.getCanonicalType();
  const clang::Type *typePtr = type.getTypePtr();

#define REGISTER_TYPE(type)                                                    \
  if (auto *node = mlir::dyn_cast<clang::type>(typePtr)) {                     \
    return convert##type(llvm::cast<clang::type>(node));                       \
  }

  REGISTER_TYPE(BuiltinType)
  REGISTER_TYPE(ArrayType)
  REGISTER_TYPE(PointerType)
  REGISTER_TYPE(TypedefType)
  REGISTER_TYPE(RecordType)

#undef REGISTER_TYPE

  llvm::errs() << "Unsupported type: " << type.getAsString();
  return nullptr;
}

mlir::Type CMLIRConverter::convertBuiltinType(const clang::BuiltinType *type) {
  mlir::OpBuilder &builder = context_manager_.Builder();

  switch (type->getKind()) {
  case clang::BuiltinType::Void:
    return builder.getNoneType();

  case clang::BuiltinType::Bool:
    return builder.getI1Type();

  case clang::BuiltinType::Char_S:
  case clang::BuiltinType::Char_U:
  case clang::BuiltinType::SChar:
  case clang::BuiltinType::UChar:
  case clang::BuiltinType::Char8:
    return builder.getI8Type();

  case clang::BuiltinType::Short:
  case clang::BuiltinType::UShort:
  case clang::BuiltinType::Char16:
    return builder.getI16Type();

  case clang::BuiltinType::Int:
  case clang::BuiltinType::UInt:
  case clang::BuiltinType::Char32:
  case clang::BuiltinType::WChar_S:
  case clang::BuiltinType::WChar_U:
    return builder.getI32Type();

  case clang::BuiltinType::Long:
  case clang::BuiltinType::ULong:
  case clang::BuiltinType::LongLong:
  case clang::BuiltinType::ULongLong:
  case clang::BuiltinType::Int128:
  case clang::BuiltinType::UInt128:
    return builder.getI64Type();

  case clang::BuiltinType::Half:
    return builder.getF16Type();

  case clang::BuiltinType::Float:
  case clang::BuiltinType::Float16:
    return builder.getF32Type();

  case clang::BuiltinType::Double:
  case clang::BuiltinType::LongDouble:
  case clang::BuiltinType::Float128:
    return builder.getF64Type();

  case clang::BuiltinType::NullPtr:
    llvm::errs() << "Warning: nullptr_t mapped to i64\n";
    return builder.getI64Type();

  default:
    llvm::errs() << "Unsupported builtin type: " << type << "\n";
  }

  return nullptr;
}

mlir::Type CMLIRConverter::convertArrayType(const clang::ArrayType *type) {
  llvm::SmallVector<int64_t, 4> dimensions;
  clang::QualType currentType = clang::QualType(type, 0);

  while (auto *arrayType =
             mlir::dyn_cast<clang::ArrayType>(currentType.getTypePtr())) {
    if (auto *constArrayType =
            mlir::dyn_cast<clang::ConstantArrayType>(arrayType)) {
      int64_t size = constArrayType->getSize().getSExtValue();
      dimensions.push_back(size);
      currentType = constArrayType->getElementType();
    } else {
      dimensions.push_back(-1);
      currentType = arrayType->getElementType();
    }
  }

  mlir::Type elementType = convertType(currentType);

  return mlir::MemRefType::get(dimensions, elementType);
}

mlir::Type CMLIRConverter::convertPointerType(const clang::PointerType *type) {
  mlir::OpBuilder &builder = context_manager_.Builder();
  clang::QualType pointeeType = type->getPointeeType().getCanonicalType();

  if (mlir::isa<clang::RecordType>(pointeeType.getTypePtr())) {
    return mlir::LLVM::LLVMPointerType::get(builder.getContext());
  }

  if (mlir::isa<clang::ArrayType>(pointeeType.getTypePtr())) {
    llvm::SmallVector<int64_t, 4> dimensions;
    clang::QualType currentType = pointeeType;

    while (auto *arrType =
               mlir::dyn_cast<clang::ArrayType>(currentType.getTypePtr())) {
      if (auto *constArrayType =
              mlir::dyn_cast<clang::ConstantArrayType>(arrType)) {
        int64_t size = constArrayType->getSize().getSExtValue();
        dimensions.push_back(size);
        currentType = constArrayType->getElementType();
      } else {
        dimensions.push_back(mlir::ShapedType::kDynamic);
        currentType = arrType->getElementType();
      }
    }

    mlir::Type elementType = convertType(currentType);
    dimensions.insert(dimensions.begin(), mlir::ShapedType::kDynamic);
    return mlir::MemRefType::get(dimensions, elementType);
  }

  mlir::Type elementType = convertType(pointeeType);
  if (!elementType)
    return nullptr;
  return mlir::MemRefType::get({mlir::ShapedType::kDynamic}, elementType);
}

mlir::Type CMLIRConverter::convertTypedefType(const clang::TypedefType *type) {
  return convertType(type->desugar());
}

mlir::Type CMLIRConverter::convertRecordType(const clang::RecordType *type) {
  mlir::OpBuilder &builder = context_manager_.Builder();
  const clang::RecordDecl *recordDecl = type->getDecl();

  auto it = recordTypeTable.find(recordDecl);
  if (it != recordTypeTable.end()) {
    return it->second;
  }

  if (!recordDecl->isCompleteDefinition()) {
    llvm::errs() << "Incomplete struct definition: "
                 << recordDecl->getNameAsString() << "\n";
    return nullptr;
  }

  llvm::SmallVector<mlir::Type, 8> memberTypes;

  for (auto *field : recordDecl->fields()) {
    mlir::Type fieldType = convertType(field->getType());
    if (!fieldType) {
      llvm::errs() << "Failed to convert field type: "
                   << field->getNameAsString() << "\n";
      return nullptr;
    }
    memberTypes.push_back(fieldType);
  }

  auto structType =
      mlir::LLVM::LLVMStructType::getLiteral(builder.getContext(), memberTypes);

  recordTypeTable[recordDecl] = structType;

  return structType;
}

// helpers
mlir::Value CMLIRConverter::convertToBool(mlir::Value value) {
  mlir::OpBuilder &builder = context_manager_.Builder();
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
