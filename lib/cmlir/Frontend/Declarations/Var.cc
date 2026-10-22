#include "../Lowering.h"
#include "clang-mlir/Casts.h"
#include "clang/AST/ExprCXX.h"

namespace cmlir::frontend {

auto ASTLowering::emitVariable(clang::VarDecl *decl) -> bool {
  if (decl->isImplicit() || llvm::isa<clang::ParmVarDecl>(decl)) {
    return true;
  }
  if (decl->isStaticLocal()) {
    context.source().error(decl->getLocation(),
                           "static local storage is unsupported");
    return false;
  }
  if (decl->hasGlobalStorage()) {
    context.source().error(decl->getLocation(),
                           "global storage is unsupported");
    return false;
  }

  auto type = types.convertObjectType(decl->getType());
  if (!type) {
    return false;
  }
  auto &builder = context.builder();
  auto loc = context.source().location(decl->getLocation());
  llvm::SmallVector<mlir::Value, 4> dynamicSizes;
  for (auto current = decl->getType(); current->isArrayType();) {
    auto *array = context.ast().getAsArrayType(current);
    if (auto *variable = llvm::dyn_cast<clang::VariableArrayType>(array)) {
      auto size = clang_mlir::toIndex(builder, loc,
                                      generateExpr(variable->getSizeExpr()));
      if (!size) {
        return false;
      }
      dynamicSizes.push_back(size);
    }
    current = array->getElementType();
  }
  auto storage =
      clang_mlir::createLocalStorage(builder, loc, type, dynamicSizes);
  if (!storage) {
    context.source().error(decl->getLocation(),
                           "unsupported local object storage");
    return false;
  }
  function.locals[decl] = storage;
  return !decl->hasInit() ||
         initializeObject(decl->getType(), decl->getInit(), storage);
}

} // namespace cmlir::frontend
