#include "ExpressionLowering.h"

namespace cmlir::frontend {

auto ExpressionLowering::generateParenExpr(clang::ParenExpr *parenExpr)
    -> mlir::Value {
  return generateExpr(parenExpr->getSubExpr());
}

} // namespace cmlir::frontend
