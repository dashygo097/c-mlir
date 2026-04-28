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
    mlir::Value value = currentFieldValueTable.lookup(fieldDecl);
    if (value) {
      return value;
    }

    value = outputValueTable.lookup(fieldDecl);
    if (value) {
      return value;
    }
  }

  llvm::WithColor::error() << "chwc: unknown decl ref: "
                           << decl->getNameAsString() << "\n";
  return nullptr;
}

} // namespace chwc
