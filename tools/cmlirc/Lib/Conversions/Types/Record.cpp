#include "../../Converter.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "clang/AST/Type.h"
#include "llvm/Support/WithColor.h"
#include <functional>

namespace cmlirc {

auto CMLIRConverter::convertRecordType(const clang::RecordType *type)
    -> mlir::Type {
  if (!type) {
    return nullptr;
  }

  mlir::OpBuilder &builder = contextManager.Builder();

  const clang::RecordDecl *recordDecl = type->getDecl();
  if (!recordDecl) {
    return nullptr;
  }

  const clang::RecordDecl *definition = recordDecl->getDefinition();
  if (!definition) {
    definition = recordDecl;
  }

  auto it = recordTypeTable.find(definition);
  if (it != recordTypeTable.end()) {
    return it->second;
  }

  if (!definition->isCompleteDefinition()) {
    llvm::WithColor::error() << "cmlirc: incomplete struct definition: "
                             << definition->getNameAsString() << "\n";
    return nullptr;
  }

  std::function<mlir::Type(clang::QualType)> convertRecordFieldType =
      [&](clang::QualType clangType) -> mlir::Type {
    if (clangType.isNull()) {
      return nullptr;
    }

    clangType = clangType.getCanonicalType();

    if (auto *arrayType =
            mlir::dyn_cast<clang::ConstantArrayType>(clangType.getTypePtr())) {
      mlir::Type elementType =
          convertRecordFieldType(arrayType->getElementType());

      if (!elementType) {
        return nullptr;
      }

      uint64_t numElements = arrayType->getSize().getZExtValue();
      return mlir::LLVM::LLVMArrayType::get(elementType, numElements);
    }

    if (mlir::isa<clang::ArrayType>(clangType.getTypePtr())) {
      llvm::WithColor::error()
          << "cmlirc: non-constant array field is not supported in record: "
          << definition->getNameAsString() << "\n";
      return nullptr;
    }

    if (clangType->isPointerType() || clangType->isReferenceType() ||
        clangType->isFunctionPointerType() || clangType->isBlockPointerType()) {
      return mlir::LLVM::LLVMPointerType::get(builder.getContext());
    }

    if (auto *nestedRecordType =
            mlir::dyn_cast<clang::RecordType>(clangType.getTypePtr())) {
      return convertRecordType(nestedRecordType);
    }

    mlir::Type fieldType = convertType(clangType);
    if (!fieldType) {
      return nullptr;
    }

    if (mlir::isa<mlir::MemRefType>(fieldType)) {
      llvm::WithColor::error()
          << "cmlirc: memref field is invalid inside LLVM struct field in "
          << definition->getNameAsString() << "\n";
      return nullptr;
    }

    return fieldType;
  };

  llvm::SmallVector<mlir::Type, 8> memberTypes;
  std::vector<const clang::FieldDecl *> fields;

  for (auto *field : definition->fields()) {
    mlir::Type fieldType = convertRecordFieldType(field->getType());
    if (!fieldType) {
      llvm::WithColor::error() << "cmlirc: failed to convert field type: "
                               << field->getNameAsString() << "\n";
      return nullptr;
    }

    memberTypes.push_back(fieldType);
    fields.push_back(field);
  }

  auto structType =
      mlir::LLVM::LLVMStructType::getLiteral(builder.getContext(), memberTypes);

  recordTypeTable[definition] = structType;
  recordFieldTable[definition] = std::move(fields);

  return structType;
}

} // namespace cmlirc
