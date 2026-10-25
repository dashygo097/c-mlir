#include "ExpressionLowering.h"
#include "clang-mlir/Constants.h"
#include "llvm/Support/WithColor.h"

namespace cmlir::frontend {

auto ExpressionLowering::generateDeclRefExpr(clang::DeclRefExpr *declRef)
    -> mlir::Value {
  mlir::OpBuilder &builder = context.builder();
  mlir::Location loc = context.source().location(declRef->getExprLoc());

  if (auto *constant =
          llvm::dyn_cast<clang::EnumConstantDecl>(declRef->getDecl())) {
    auto type = mlir::dyn_cast_or_null<mlir::IntegerType>(
        types.convertType(constant->getCanonicalDecl()->getType()));
    return clang_mlir::intConst(builder, loc, type, constant->getInitVal());
  }

  if (auto *varDecl = mlir::dyn_cast<clang::VarDecl>(declRef->getDecl())) {
    if (auto *parmDecl = mlir::dyn_cast<clang::ParmVarDecl>(varDecl)) {
      if (function.parameters.count(parmDecl)) {
        return function.parameters[parmDecl];
      }
    }

    if (function.locals.count(varDecl)) {
      return declRef->getType()->isArrayType() ? function.locals[varDecl]
                                               : loadLValue(declRef);
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
