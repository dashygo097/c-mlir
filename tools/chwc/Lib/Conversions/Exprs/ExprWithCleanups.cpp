#include "../../Converter.h"

namespace chwc {

auto CHWConverter::generateExprWithCleanups(clang::ExprWithCleanups *expr)
    -> mlir::Value {
  if (!expr) {
    return nullptr;
  }

  return generateExpr(expr->getSubExpr());
}

} // namespace chwc
