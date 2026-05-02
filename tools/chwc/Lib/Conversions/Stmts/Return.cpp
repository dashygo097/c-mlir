#include "../../Converter.h"

namespace chwc {

auto CHWConverter::generateReturnStmt(clang::ReturnStmt *returnStmt)
    -> mlir::Value {
  if (!returnStmt) {
    return nullptr;
  }

  clang::Expr *retValue = returnStmt->getRetValue();
  if (!retValue) {
    return nullptr;
  }

  return generateExpr(retValue);
}

} // namespace chwc
