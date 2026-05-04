#include "../../Converter.h"

namespace chwc {

auto CHWConverter::generateMaterializeTemporaryExpr(
    clang::MaterializeTemporaryExpr *expr) -> mlir::Value {
  if (!expr) {
    return nullptr;
  }

  return generateExpr(expr->getSubExpr());
}

} // namespace chwc
