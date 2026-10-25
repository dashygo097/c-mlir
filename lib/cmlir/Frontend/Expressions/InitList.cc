#include "../Initialization.h"
#include "ExpressionLowering.h"

namespace cmlir::frontend {

auto ExpressionLowering::generateInitListExpr(clang::InitListExpr *list)
    -> mlir::Value {
  auto type = types.convertObjectType(list->getType());
  auto &builder = context.builder();
  auto loc = context.source().location(list->getExprLoc());
  auto storage = clang_mlir::createLocalStorage(builder, loc, type);
  if (!storage || !ObjectInitialization(context, types, *this)
                       .emit(list->getType(), list, storage)) {
    return {};
  }
  if (list->getType()->isArrayType()) {
    return storage;
  }
  auto address = clang_mlir::Address::get(storage, type);
  return mlir::succeeded(address) ? address->load(builder, loc) : mlir::Value{};
}

} // namespace cmlir::frontend
