#include "../../Converter.h"
#include "llvm/Support/WithColor.h"

namespace chwc {

auto CHWConverter::generateDeclRefExpr(clang::DeclRefExpr *declRef)
    -> mlir::Value {
  clang::ValueDecl *decl = declRef->getDecl();

  if (auto *varDecl = mlir::dyn_cast<clang::VarDecl>(decl)) {
    mlir::Value value = localValueTable.lookup(varDecl);
    if (value) {
      return value;
    }
  }

  if (auto *fieldDecl = mlir::dyn_cast<clang::FieldDecl>(decl)) {
    auto fieldIt = fieldTable.find(fieldDecl);
    if (fieldIt == fieldTable.end()) {
      llvm::WithColor::error()
          << "chwc: unknown field ref: " << fieldDecl->getNameAsString()
          << "\n";
      return nullptr;
    }

    HWFieldInfo &fieldInfo = fieldIt->second;

    if (fieldInfo.kind == HWFieldKind::Output) {
      mlir::Value value = outputValueTable.lookup(fieldDecl);
      if (value) {
        return value;
      }
    }

    if (fieldInfo.kind == HWFieldKind::Reg ||
        fieldInfo.kind == HWFieldKind::Wire) {
      mlir::Value value = nextFieldValueTable.lookup(fieldDecl);
      if (value) {
        return value;
      }
    }

    mlir::Value value = currentFieldValueTable.lookup(fieldDecl);
    if (value) {
      return value;
    }

    llvm::WithColor::error()
        << "chwc: hardware field has no value: " << fieldInfo.name << "\n";
    return nullptr;
  }

  llvm::WithColor::error() << "chwc: unknown decl ref: "
                           << decl->getNameAsString() << "\n";
  return nullptr;
}

} // namespace chwc
