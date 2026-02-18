#include "../../Converter.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"

namespace cmlirc {

std::optional<unsigned>
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
  for (unsigned i = 0; i < fields.size(); ++i) {
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
    llvm::errs() << "Failed to generate base expression\n";
    return nullptr;
  }

  clang::FieldDecl *fieldDecl =
      llvm::dyn_cast<clang::FieldDecl>(memberExpr->getMemberDecl());
  if (!fieldDecl) {
    llvm::errs() << "Member is not a field\n";
    return nullptr;
  }

  clang::QualType baseType = base->getType();

  if (memberExpr->isArrow()) {
    baseType = baseType->getPointeeType();
  }

  const clang::RecordType *recordType = baseType->getAsStructureType();
  if (!recordType) {
    llvm::errs() << "Base is not a struct type\n";
    return nullptr;
  }

  const clang::RecordDecl *recordDecl = recordType->getDecl();

  auto fieldIndexOpt = getFieldIndex(recordDecl, fieldDecl);
  if (!fieldIndexOpt) {
    llvm::errs() << "Field not found in struct\n";
    return nullptr;
  }
  unsigned fieldIndex = *fieldIndexOpt;

  // 方案1：如果 struct 存储在 memref 中，使用指针运算
  if (auto memrefType = mlir::dyn_cast<mlir::MemRefType>(baseValue.getType())) {
    // struct 存储在内存中
    // 需要计算字段偏移量，这里简化处理

    // 对于 LLVM struct，使用 llvm.getelementptr
    // 但在 memref 中，我们需要先转换

    llvm::errs() << "Warning: struct in memref not fully supported yet\n";
    return nullptr;
  }

  if (auto structType =
          mlir::dyn_cast<mlir::LLVM::LLVMStructType>(baseValue.getType())) {
    auto fieldValue =
        mlir::LLVM::ExtractValueOp::create(builder, loc, baseValue, fieldIndex);
    return fieldValue;
  }

  llvm::errs() << "Unsupported base type for member access\n";
  return nullptr;
}

} // namespace cmlirc
