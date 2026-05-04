#include "../../Converter.h"

namespace chwc {

auto CHWConverter::generateCXXBindTemporaryExpr(
    clang::CXXBindTemporaryExpr *expr) -> mlir::Value {
  if (!expr) {
    return nullptr;
  }

  return generateExpr(expr->getSubExpr());
}

} // namespace chwc
