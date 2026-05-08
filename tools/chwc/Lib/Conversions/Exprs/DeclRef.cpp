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

    HWFieldInfo &fieldInfo = fieldIt->second;

    switch (fieldInfo.kind) {
    case HWFieldKind::Input:
    case HWFieldKind::Wire:
    case HWFieldKind::Reg: {
      mlir::Value value = moduleContext.currentValues.lookup(fieldDecl);
      if (!value) {
        llvm::WithColor::error()
            << "chwc: hardware field is not wired yet: " << fieldInfo.name
            << "\n";
        return nullptr;
      }

      return value;
    }

    case HWFieldKind::Output: {
      mlir::Value value = moduleContext.outputValues.lookup(fieldDecl);
      if (!value) {
        llvm::WithColor::error()
            << "chwc: output field is read before assignment: "
            << fieldInfo.name << "\n";
        return nullptr;
      }

      return value;
    }
    }
  }

  llvm::WithColor::error() << "chwc: unsupported DeclRefExpr: "
                           << declRef->getDecl()->getNameAsString() << "\n";
  return nullptr;
}

} // namespace chwc
