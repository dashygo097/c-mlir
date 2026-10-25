#include "../Initialization.h"
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

  if (decl->getType()->isReferenceType()) {
    auto address = expressions.emitAddress(*decl->getInit());
    if (mlir::failed(address)) {
      return false;
    }
    function.locals[decl] = address->materialize(
        context.builder(), context.source().location(decl->getLocation()));
    return static_cast<bool>(function.locals[decl]);
  }
  if (decl->getType()->isPointerType() && decl->hasInit()) {
    auto initial = expressions.emitValue(*decl->getInit());
    if (mlir::failed(initial)) {
      return false;
    }
    auto &builder = context.builder();
    auto loc = context.source().location(decl->getLocation());
    auto value = *initial;
    if (mlir::isa<mlir::MemRefType>(value.getType())) {
      value = clang_mlir::toValue(builder, loc, value,
                                  types.convertType(decl->getType()));
    }
    function.locals[decl] = clang_mlir::createLocalSlot(builder, loc, value);
    return static_cast<bool>(function.locals[decl]);
  }
  auto type = types.convertObjectType(decl->getType());
  if (decl->getType()->isPointerType()) {
    auto valueType = types.convertType(decl->getType());
    if (valueType && mlir::MemRefType::isValidElementType(valueType)) {
      type = mlir::MemRefType::get({}, valueType);
    }
  }
  if (!type) {
    return false;
  }
  auto &builder = context.builder();
  auto loc = context.source().location(decl->getLocation());
  llvm::SmallVector<mlir::Value, 4> dynamicSizes;
  for (auto current = decl->getType(); current->isArrayType();) {
    auto *array = context.ast().getAsArrayType(current);
    if (auto *variable = llvm::dyn_cast<clang::VariableArrayType>(array)) {
      auto value = expressions.emitValue(*variable->getSizeExpr());
      if (mlir::failed(value)) {
        return false;
      }
      auto size = clang_mlir::toIndex(builder, loc, *value);
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
         ObjectInitialization(context, types, expressions)
             .emit(decl->getType(), decl->getInit(), storage);
}

} // namespace cmlir::frontend
