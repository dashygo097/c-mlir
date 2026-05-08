#include "../../Converter.h"
#include "../Utils/Cast.h"

namespace chwc {

auto CHWConverter::generateCXXFunctionalCastExpr(
    clang::CXXFunctionalCastExpr *castExpr) -> mlir::Value {
  if (!castExpr) {
    return nullptr;
  }

  mlir::Value value = generateExpr(castExpr->getSubExpr());
  if (!value) {
    return nullptr;
  }

  mlir::Type targetType = convertType(castExpr->getType());
  if (!targetType) {
    return value;
  }

  mlir::OpBuilder &builder = contextManager.Builder();
  mlir::Location loc = builder.getUnknownLoc();

  return utils::promoteValue(builder, loc, value, targetType);
}

} // namespace chwc
