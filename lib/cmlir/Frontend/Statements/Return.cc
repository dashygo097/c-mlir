#include "../Lowering.h"
#include "clang-mlir/Casts.h"

namespace cmlir::frontend {

auto ASTLowering::emitReturn(clang::ReturnStmt *stmt) -> bool {
  auto &builder = context.builder();
  auto loc = context.source().location(stmt->getReturnLoc());
  auto results = function.operation.getFunctionType().getResults();
  mlir::Value value;
  if (auto *expr = stmt->getRetValue()) {
    value = generateExpr(expr);
    if (!results.empty()) {
      if (!value) {
        return false;
      }
      value = clang_mlir::toValue(
          builder, loc, value, results.front(),
          expr->getType()->isSignedIntegerOrEnumerationType());
      if (!value) {
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
