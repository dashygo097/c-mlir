#include "../../Lowering.h"
#include "clang-mlir/FloatingPoint.h"
#include "clang-mlir/Numerics.h"

namespace cmlir::frontend {

auto ASTLowering::generateIncDecUnaryOperator(clang::UnaryOperator *expr)
    -> mlir::Value {
  auto address = emitAddress(expr->getSubExpr());
  if (mlir::failed(address)) {
    return {};
  }
  auto &builder = context.builder();
  auto loc = context.source().location(expr->getExprLoc());
  auto old = address->load(builder, loc);
  auto flags = clang_mlir::getFastMathFlags(*expr, context.ast().getLangOpts());
  auto value = expr->isIncrementOp()
                   ? clang_mlir::inc(builder, loc, old, flags)
                   : clang_mlir::dec(builder, loc, old, flags);
  if (!value || mlir::failed(address->store(builder, loc, value))) {
    return {};
  }
  return expr->isPrefix() ? value : old;
}

} // namespace cmlir::frontend
