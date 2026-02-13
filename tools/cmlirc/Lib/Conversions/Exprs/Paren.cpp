#include "../../Converter.h"

namespace cmlirc {

mlir::Value CMLIRConverter::generateParenExpr(clang::ParenExpr *parenExpr) {
  return generateExpr(parenExpr->getSubExpr());
}

} // namespace cmlirc
