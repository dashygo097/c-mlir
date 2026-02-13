#include "../../Converter.h"

namespace cmlirc {
mlir::Value CMLIRConverter::generateDeclRefExpr(clang::DeclRefExpr *declRef) {
  if (auto *varDecl = llvm::dyn_cast<clang::VarDecl>(declRef->getDecl())) {
    if (auto *parmDecl = llvm::dyn_cast<clang::ParmVarDecl>(varDecl)) {
      if (paramTable.count(parmDecl)) {
        return paramTable[parmDecl];
      }
    }

    if (symbolTable.count(varDecl)) {
      return symbolTable[varDecl];
    }

    llvm::errs() << "Variable not found: " << varDecl->getName() << "\n";
    return nullptr;
  }

  if (auto *funcDecl =
          llvm::dyn_cast<clang::FunctionDecl>(declRef->getDecl())) {
    if (functionTable.count(funcDecl)) {
      return functionTable[funcDecl];
    }

    llvm::errs() << "Function not found: " << funcDecl->getName() << "\n";
    return nullptr;
  }

  llvm::errs() << "Unsupported DeclRefExpr type: "
               << declRef->getDecl()->getDeclKindName() << "\n";
  return nullptr;
}
} // namespace cmlirc
