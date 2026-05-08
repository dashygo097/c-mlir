#include "../../Converter.h"
#include "../Utils/Constant.h"

namespace chwc {

auto CHWConverter::generateCXXBoolLiteralExpr(
    clang::CXXBoolLiteralExpr *boolLit) -> mlir::Value {
  mlir::OpBuilder &builder = contextManager.Builder();
  mlir::Location loc = builder.getUnknownLoc();

  return utils::boolConst(builder, loc, boolLit->getValue());
}

auto CHWConverter::generateIntegerLiteral(clang::IntegerLiteral *intLit)
    -> mlir::Value {
  mlir::OpBuilder &builder = contextManager.Builder();
  mlir::Location loc = builder.getUnknownLoc();

  mlir::Type type = convertType(intLit->getType());
  if (!type) {
    type = builder.getIntegerType(32);
  }

  return utils::intConst(builder, loc, type, intLit->getValue().getSExtValue());
}

} // namespace chwc
