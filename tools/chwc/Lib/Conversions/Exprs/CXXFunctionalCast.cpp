#include "../../Converter.h"
#include "../Utils/Cast.h"
#include "../Utils/Constant.h"
#include "llvm/Support/WithColor.h"

namespace chwc {

auto CHWConverter::generateCXXFunctionalCastExpr(
    clang::CXXFunctionalCastExpr *expr) -> mlir::Value {
  if (!expr) {
    return nullptr;
  }

  mlir::OpBuilder &builder = contextManager.Builder();
  mlir::Location loc = builder.getUnknownLoc();

  mlir::Type targetType = convertType(expr->getType());
  if (!targetType) {
    return nullptr;
  }

  clang::Expr *subExpr = expr->getSubExpr();
  if (!subExpr) {
    return utils::zeroValue(builder, loc, targetType);
  }

  mlir::Value value = generateExpr(subExpr);
  if (!value) {
    return nullptr;
  }

  return utils::promoteValue(builder, loc, value, targetType);
}

} // namespace chwc
