#include "../../Converter.h"
#include "../Utils/Constants.h"

namespace chwc {

auto CHWConverter::generateCXXBoolLiteralExpr(
    clang::CXXBoolLiteralExpr *boolLit) -> mlir::Value {
  mlir::OpBuilder &builder = contextManager.Builder();
  mlir::Location loc = builder.getUnknownLoc();

  return utils::intConst(builder, loc, builder.getI1Type(),
                         boolLit->getValue() ? 1 : 0);
}

auto CHWConverter::generateIntegerLiteral(clang::IntegerLiteral *intLit)
    -> mlir::Value {
  mlir::OpBuilder &builder = contextManager.Builder();
  mlir::Location loc = builder.getUnknownLoc();

  return utils::intConst(builder, loc, builder.getI32Type(),
                         intLit->getValue().getLimitedValue());
}

} // namespace chwc
