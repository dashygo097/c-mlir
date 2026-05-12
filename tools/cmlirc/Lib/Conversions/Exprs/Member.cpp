#include "../../Converter.h"
#include "../Utils/Constants.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "clang/AST/Expr.h"
#include "llvm/Support/WithColor.h"

namespace cmlirc {

auto CMLIRConverter::getFieldIndex(const clang::RecordDecl *recordDecl,
                                   const clang::FieldDecl *fieldDecl)
    -> std::optional<uint32_t> {
  if (!recordDecl || !fieldDecl) {
    return std::nullopt;
  }

  const clang::RecordDecl *definition = recordDecl->getDefinition();
  if (!definition) {
    definition = recordDecl;
  }

  auto it = recordFieldTable.find(definition);
  if (it == recordFieldTable.end()) {
    std::vector<const clang::FieldDecl *> fields;
    for (auto *field : definition->fields()) {
      fields.push_back(field);
    }

    recordFieldTable[definition] = std::move(fields);
    it = recordFieldTable.find(definition);
  }

  const clang::FieldDecl *canonicalField = fieldDecl->getCanonicalDecl();

  const auto &fields = it->second;
  for (uint32_t i = 0; i < fields.size(); ++i) {
    if (fields[i] == fieldDecl ||
        fields[i]->getCanonicalDecl() == canonicalField) {
      return i;
    }
  }

  return std::nullopt;
}

auto CMLIRConverter::generateMemberExpr(clang::MemberExpr *memberExpr)
    -> mlir::Value {
  mlir::OpBuilder &builder = contextManager.Builder();
  mlir::Location loc = builder.getUnknownLoc();

  clang::Expr *base = memberExpr->getBase();
  mlir::Value baseValue = generateExpr(base);
  if (!baseValue) {
    return nullptr;
  }

  auto *fieldDecl =
      mlir::dyn_cast<clang::FieldDecl>(memberExpr->getMemberDecl());
  if (!fieldDecl) {
    llvm::WithColor::error() << "cmlirc: Member is not a field\n";
    return nullptr;
  }

  clang::QualType baseType = base->getType().getCanonicalType();

  if (memberExpr->isArrow()) {
    if (!baseType->isPointerType()) {
      llvm::WithColor::error()
          << "cmlirc: arrow member base is not pointer type\n";
      return nullptr;
    }

    baseType = baseType->getPointeeType().getCanonicalType();
  }

  const auto *recordType = baseType->getAs<clang::RecordType>();
  if (!recordType) {
    llvm::WithColor::error() << "cmlirc: Base is not a struct or class type\n";
    return nullptr;
  }

  const clang::RecordDecl *recordDecl = recordType->getDecl();
  const clang::RecordDecl *definition = recordDecl->getDefinition();
  if (!definition) {
    definition = recordDecl;
  }

  std::optional<uint32_t> fieldIndexOpt = getFieldIndex(definition, fieldDecl);
  if (!fieldIndexOpt) {
    llvm::WithColor::error() << "cmlirc: field not found in struct/class\n";
    return nullptr;
  }

  uint32_t fieldIndex = *fieldIndexOpt;

  mlir::Type structType = convertType(baseType);
  auto llvmStructType =
      mlir::dyn_cast_or_null<mlir::LLVM::LLVMStructType>(structType);
  if (!llvmStructType) {
    llvm::WithColor::error() << "cmlirc: expected LLVM struct type\n";
    return nullptr;
  }

  if (lastArrayAccess && lastArrayAccess->base == baseValue) {
    lastArrayAccess.reset();
  }

  if (mlir::isa<mlir::LLVM::LLVMPointerType>(baseValue.getType())) {
    auto ptrType = mlir::LLVM::LLVMPointerType::get(builder.getContext());

    llvm::SmallVector<mlir::Value, 2> indices;
    indices.push_back(utils::intConst(builder, loc, builder.getI32Type(), 0));
    indices.push_back(
        utils::intConst(builder, loc, builder.getI32Type(), fieldIndex));

    auto fieldPtr = mlir::LLVM::GEPOp::create(
        builder, loc, ptrType, llvmStructType, baseValue, indices);

    return fieldPtr.getResult();
  }

  if (mlir::isa<mlir::LLVM::LLVMStructType>(baseValue.getType())) {
    auto fieldValue =
        mlir::LLVM::ExtractValueOp::create(builder, loc, baseValue, fieldIndex);
    return fieldValue.getResult();
  }

  llvm::WithColor::error()
      << "cmlirc: unsupported base type for member access\n";
  return nullptr;
}

} // namespace cmlirc
