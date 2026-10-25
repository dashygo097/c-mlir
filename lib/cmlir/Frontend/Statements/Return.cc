#include "../Lowering.h"
#include "clang-mlir/Casts.h"

namespace cmlir::frontend {

auto ASTLowering::emitReturn(clang::ReturnStmt *stmt) -> bool {
  auto &builder = context.builder();
  auto loc = context.source().location(stmt->getReturnLoc());
  auto results = function.operation.getFunctionType().getResults();
  mlir::Value value;
  if (auto *expr = stmt->getRetValue()) {
    if (results.empty()) {
      if (mlir::failed(expressions.emitDiscarded(*expr))) {
        return false;
      }
    } else if (function.declaration->getReturnType()->isReferenceType()) {
      auto address = expressions.emitAddress(*expr);
      if (mlir::failed(address)) {
        return false;
      }
      value = address->materialize(builder, loc);
    } else {
      auto emitted = expressions.emitValue(*expr);
      if (mlir::failed(emitted)) {
        return false;
      }
      value = *emitted;
    }
    if (!results.empty()) {
      if (!value) {
        return false;
      }
      value = clang_mlir::toValue(
          builder, loc, value, results.front(),
          expr->getType()->isSignedIntegerOrEnumerationType());
      if (!value) {
        context.source().error(expr->getExprLoc(),
                               "cannot convert function return value");
        return false;
      }
    }
  }
  if (!results.empty() && !value) {
    context.source().error(stmt->getReturnLoc(),
                           "missing function return value");
    return false;
  }
  mlir::func::ReturnOp::create(builder, loc,
                               results.empty() ? mlir::ValueRange{}
                                               : mlir::ValueRange{value});
  return true;
}

} // namespace cmlir::frontend
