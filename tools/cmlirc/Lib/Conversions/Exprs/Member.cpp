#include "../../Converter.h"
#include "../Utils/Constants.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "llvm/Support/WithColor.h"

namespace cmlirc {

std::optional<uint32_t>
CMLIRConverter::getFieldIndex(const clang::RecordDecl *recordDecl,
                              const clang::FieldDecl *fieldDecl) {
  auto it = recordFieldTable.find(recordDecl);
  if (it == recordFieldTable.end()) {
    std::vector<const clang::FieldDecl *> fields;
    for (auto *field : recordDecl->fields()) {
      fields.push_back(field);
    }
    recordFieldTable[recordDecl] = fields;
    it = recordFieldTable.find(recordDecl);
  }

  const auto &fields = it->second;
  for (uint32_t i = 0; i < fields.size(); ++i) {
    if (fields[i] == fieldDecl) {
      return i;
    }
  }

  return std::nullopt;
}

mlir::Value CMLIRConverter::generateMemberExpr(clang::MemberExpr *memberExpr) {
  mlir::OpBuilder &builder = context_manager_.Builder();
  mlir::Location loc = builder.getUnknownLoc();

  clang::Expr *base = memberExpr->getBase();
  mlir::Value baseValue = generateExpr(base);
  if (!baseValue) {
    llvm::WithColor::error() << "cmlirc: failed to generate base expression\n";
    return nullptr;
  }

  clang::FieldDecl *fieldDecl =
      mlir::dyn_cast<clang::FieldDecl>(memberExpr->getMemberDecl());
  if (!fieldDecl) {
    llvm::WithColor::error() << "cmlirc: Member is not a field\n";
    return nullptr;
  }

  clang::QualType baseType = base->getType();

  if (memberExpr->isArrow()) {
    baseType = baseType->getPointeeType();
  }

  const clang::RecordType *recordType = baseType->getAsStructureType();
  if (!recordType) {
    llvm::WithColor::error() << "cmlirc: Base is not a struct type\n";
    return nullptr;
  }

  const clang::RecordDecl *recordDecl = recordType->getDecl();

  std::optional<uint32_t> fieldIndexOpt = getFieldIndex(recordDecl, fieldDecl);
  if (!fieldIndexOpt) {
    llvm::WithColor::error() << "cmlirc: field not found in struct\n";
    return nullptr;
  }
  uint32_t fieldIndex = *fieldIndexOpt;

  mlir::Type structType = convertType(baseType);
  auto llvmStructType = mlir::dyn_cast<mlir::LLVM::LLVMStructType>(structType);
  if (!llvmStructType) {
    llvm::WithColor::error() << "cmlirc: expected LLVM struct type\n";
    return nullptr;
  }

  if (mlir::isa<mlir::LLVM::LLVMPointerType>(baseValue.getType())) {
    auto ptrType = mlir::LLVM::LLVMPointerType::get(builder.getContext());

    llvm::SmallVector<mlir::Value, 2> indices;
    indices.push_back(detail::intConst(builder, loc, builder.getI32Type(), 0));
    indices.push_back(
        detail::intConst(builder, loc, builder.getI32Type(), fieldIndex));

    auto fieldPtr = mlir::LLVM::GEPOp::create(
        builder, loc, ptrType, llvmStructType, baseValue, indices);

    return fieldPtr.getResult();
  }

  if (mlir::isa<mlir::LLVM::LLVMStructType>(baseValue.getType())) {
    auto fieldValue =
        mlir::LLVM::ExtractValueOp::create(builder, loc, baseValue, fieldIndex);
    return fieldValue;
  }

  llvm::WithColor::error()
      << "cmlirc: unsupported base type for member access\n";
  return nullptr;
}

} // namespace cmlirc
