#include "TypeLowering.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "clang/AST/Attr.h"
#include "clang/AST/DeclCXX.h"
#include "clang/AST/Type.h"
#include "llvm/Support/WithColor.h"
#include <functional>

namespace cmlir::frontend {

auto TypeLowering::convertRecordType(const clang::RecordType *type)
    -> mlir::Type {
  if (!type) {
    return nullptr;
  }

  mlir::OpBuilder &builder = context.builder();

  const clang::RecordDecl *recordDecl = type->getDecl();
  if (!recordDecl) {
    return nullptr;
  }

  const clang::RecordDecl *definition = recordDecl->getDefinition();
  if (!definition) {
    definition = recordDecl;
  }

  if (definition->isUnion() || definition->hasAttr<clang::PackedAttr>() ||
      definition->hasAttr<clang::AlignedAttr>()) {
    context.source().error(definition->getLocation(),
                           "union and custom record layouts are unsupported");
    return {};
  }
  if (auto *record = llvm::dyn_cast<clang::CXXRecordDecl>(definition)) {
    if (record->getNumBases() != 0) {
      context.source().error(record->getLocation(),
                             "record inheritance is unsupported");
      return {};
    }
  }
  auto it = recordTypeTable.find(definition);
  if (it != recordTypeTable.end()) {
    return it->second;
  }

  if (!definition->isCompleteDefinition()) {
    llvm::WithColor::error() << "clang-to-mlir: incomplete struct definition: "
                             << definition->getNameAsString() << "\n";
    return nullptr;
  }

  llvm::SmallVector<mlir::Type, 8> memberTypes;
  std::vector<const clang::FieldDecl *> fields;

  for (auto *field : definition->fields()) {
    if (field->isBitField()) {
      context.source().error(field->getLocation(),
                             "bitfield layouts are unsupported");
      return {};
    }
    mlir::Type fieldType = convertRecordFieldType(field->getType());
    if (!fieldType) {
      llvm::WithColor::error()
          << "clang-to-mlir: failed to convert field type: "
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

auto TypeLowering::convertRecordFieldType(clang::QualType clangType)
    -> mlir::Type {
  auto &builder = context.builder();
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
    llvm::WithColor::error() << "clang-to-mlir: non-constant array field is "
                                "not supported in record: "
                             << "\n";
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
        << "clang-to-mlir: memref field is invalid inside LLVM struct field in "
        << "\n";
    return nullptr;
  }

  return fieldType;
}

} // namespace cmlir::frontend
