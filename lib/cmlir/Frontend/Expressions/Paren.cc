#include "../Lowering.h"

namespace cmlir::frontend {

auto ASTLowering::generateParenExpr(clang::ParenExpr *parenExpr)
    -> mlir::Value {
  return generateExpr(parenExpr->getSubExpr());
}

} // namespace cmlir::frontend
