#include "../../Converter.h"
#include "../Utils/Cast.h"
#include "../Utils/Constant.h"

namespace chwc {

auto CHWConverter::generateCXXFunctionalCastExpr(
    clang::CXXFunctionalCastExpr *castExpr) -> mlir::Value {
  if (!castExpr) {
    return nullptr;
  }

  mlir::OpBuilder &builder = contextManager.Builder();
  mlir::Location loc = builder.getUnknownLoc();

  mlir::Type targetType = convertType(castExpr->getType());
  if (!targetType) {
    return nullptr;
  }

  clang::Expr *subExpr = castExpr->getSubExpr()->IgnoreParenImpCasts();

  if (auto *intLit = mlir::dyn_cast<clang::IntegerLiteral>(subExpr)) {
    return utils::intConst(builder, loc, targetType,
                           intLit->getValue().getSExtValue());
  }

  mlir::Value value = generateExpr(subExpr);
  if (!value) {
    return nullptr;
  }

  if (value.getType() == targetType) {
    return value;
  }

  return utils::promoteValue(builder, loc, value, targetType);
}

} // namespace chwc
