#include "../../Converter.h"
#include "llvm/Support/WithColor.h"

namespace cmlirc {
auto CMLIRConverter::generateDeclRefExpr(clang::DeclRefExpr *declRef)
    -> mlir::Value {
  if (auto *varDecl = mlir::dyn_cast<clang::VarDecl>(declRef->getDecl())) {
    if (auto *parmDecl = mlir::dyn_cast<clang::ParmVarDecl>(varDecl)) {
      if (paramTable.count(parmDecl)) {
        return paramTable[parmDecl];
      }
    }
    if (symbolTable.count(varDecl)) {
      mlir::Value val = symbolTable[varDecl];
      if (varDecl->getType()->isPointerType()) {
        return val;
      }
      return val;
    }
    llvm::WithColor::error()
        << "cmlirc: variable not found: " << varDecl->getName() << "\n";
    return nullptr;
  }

  if (auto *funcDecl =
          mlir::dyn_cast<clang::FunctionDecl>(declRef->getDecl())) {
    if (functionTable.count(funcDecl)) {
      return functionTable[funcDecl];
    }

    llvm::WithColor::error()
        << "cmlirc: function not found: " << funcDecl->getName() << "\n";
    return nullptr;
  }

  llvm::WithColor::error() << "cmlirc: unsupported DeclRefExpr type: "
                           << declRef->getDecl()->getDeclKindName() << "\n";
  return nullptr;
}
} // namespace cmlirc
