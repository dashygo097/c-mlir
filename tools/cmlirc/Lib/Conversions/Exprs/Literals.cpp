#include "../../Converter.h"
#include "../Types/Types.h"

namespace cmlirc {

mlir::Value
CMLIRConverter::generateCXXBoolLiteralExpr(clang::CXXBoolLiteralExpr *boolLit) {
  mlir::OpBuilder &builder = context_manager_.Builder();
  mlir::Location loc = builder.getUnknownLoc();

  bool value = boolLit->getValue();
  mlir::Type type = convertType(builder, boolLit->getType());

  return mlir::arith::ConstantOp::create(
             builder, loc, type, builder.getIntegerAttr(type, value ? 1 : 0))
      .getResult();
}

mlir::Value
CMLIRConverter::generateIntegerLiteral(clang::IntegerLiteral *intLit) {
  mlir::OpBuilder &builder = context_manager_.Builder();
  mlir::Location loc = builder.getUnknownLoc();

  int64_t value = intLit->getValue().getSExtValue();
  mlir::Type type = convertType(builder, intLit->getType());

  return mlir::arith::ConstantOp::create(builder, loc, type,
                                         builder.getIntegerAttr(type, value))
      .getResult();
}

mlir::Value
CMLIRConverter::generateFloatingLiteral(clang::FloatingLiteral *floatLit) {
  mlir::OpBuilder &builder = context_manager_.Builder();
  mlir::Location loc = builder.getUnknownLoc();

  auto value = floatLit->getValue();
  mlir::Type type = convertType(builder, floatLit->getType());

  return mlir::arith::ConstantOp::create(builder, loc, type,
                                         builder.getFloatAttr(type, value))
      .getResult();
}
} // namespace cmlirc
