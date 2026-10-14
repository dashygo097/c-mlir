#include "../../Lowering.h"
#include "clang-mlir/Numerics.h"

namespace cmlir::frontend {

auto ASTLowering::generateIncDecUnaryOperator(clang::Expr *expr, bool increment,
                                              bool prefix) -> mlir::Value {
  auto address = emitAddress(expr);
  if (mlir::failed(address)) {
    return {};
  }
  auto &builder = context.builder();
  auto loc = context.source().location(expr->getExprLoc());
  auto old = address->load(builder, loc);
  auto value = increment ? clang_mlir::inc(builder, loc, old)
                         : clang_mlir::dec(builder, loc, old);
  if (!value || mlir::failed(address->store(builder, loc, value))) {
    return {};
  }
  return prefix ? value : old;
}

} // namespace cmlir::frontend
