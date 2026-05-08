#include "../../Converter.h"
#include "llvm/Support/WithColor.h"

namespace chwc {

auto CHWConverter::generateDeclRefExpr(clang::DeclRefExpr *declRef)
    -> mlir::Value {
  if (!declRef) {
    return nullptr;
  }

  if (auto *varDecl = mlir::dyn_cast<clang::VarDecl>(declRef->getDecl())) {
    if (!functionStack.empty()) {
      mlir::Value value = functionStack.back().locals.lookup(varDecl);
      if (value) {
        return value;
      }
    }
  }

  if (auto *fieldDecl = mlir::dyn_cast<clang::FieldDecl>(declRef->getDecl())) {
    auto fieldIt = moduleContext.fields.find(fieldDecl);
    if (fieldIt == moduleContext.fields.end()) {
      llvm::WithColor::error()
          << "chwc: unknown hardware field: " << fieldDecl->getNameAsString()
          << "\n";
      return nullptr;
    }

    if (fieldIt->second.kind == HWFieldKind::Output) {
      llvm::WithColor::error()
          << "chwc: reading output field is not supported: "
          << fieldIt->second.name << "\n";
      return nullptr;
    }

    mlir::Value value = moduleContext.currentValues.lookup(fieldDecl);
    if (!value) {
      llvm::WithColor::error()
          << "chwc: hardware field is not wired yet: " << fieldIt->second.name
          << "\n";
      return nullptr;
    }

    return value;
  }

  llvm::WithColor::error() << "chwc: unsupported DeclRefExpr: "
                           << declRef->getDecl()->getNameAsString() << "\n";
  return nullptr;
}

} // namespace chwc
