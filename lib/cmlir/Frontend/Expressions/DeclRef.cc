#include "../Lowering.h"
#include "llvm/Support/WithColor.h"

namespace cmlir::frontend {

auto ASTLowering::generateDeclRefExpr(clang::DeclRefExpr *declRef)
    -> mlir::Value {
  if (auto *varDecl = mlir::dyn_cast<clang::VarDecl>(declRef->getDecl())) {
    if (auto *parmDecl = mlir::dyn_cast<clang::ParmVarDecl>(varDecl)) {
      if (function.parameters.count(parmDecl)) {
        return function.parameters[parmDecl];
      }
    }

    if (function.locals.count(varDecl)) {
      return function.locals[varDecl];
    }

    llvm::WithColor::error()
        << "clang-to-mlir: variable not found: " << varDecl->getName() << "\n";
    return nullptr;
  }

  llvm::WithColor::error() << "clang-to-mlir: unsupported DeclRefExpr type: "
                           << declRef->getDecl()->getDeclKindName() << "\n";
  return nullptr;
}

} // namespace cmlir::frontend
