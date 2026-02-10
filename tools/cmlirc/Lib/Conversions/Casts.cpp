#include "../ASTVisitor.h"

namespace cmlirc {

mlir::Value CMLIRCASTVisitor::convertToBool(mlir::Value value) {
  mlir::OpBuilder &builder = context_manager_.Builder();
  mlir::Type type = value.getType();

  if (type.isInteger(1)) {
    return value;
  }

  if (auto intType = mlir::dyn_cast<mlir::IntegerType>(type)) {
    mlir::Value zero =
        mlir::arith::ConstantOp::create(builder, builder.getUnknownLoc(), type,
                                        builder.getIntegerAttr(type, 0));
    return mlir::arith::CmpIOp::create(builder, builder.getUnknownLoc(),
                                       mlir::arith::CmpIPredicate::ne, value,
                                       zero)
        .getResult();
  }

  if (auto floatType = mlir::dyn_cast<mlir::FloatType>(type)) {
    mlir::Value zero =
        mlir::arith::ConstantOp::create(builder, builder.getUnknownLoc(), type,
                                        builder.getFloatAttr(type, 0.0));
    return mlir::arith::CmpFOp::create(
               builder, builder.getUnknownLoc(),
               mlir::arith::CmpFPredicate::ONE, // ONE = Ordered Not Equal
               value, zero)
        .getResult();
  }

  llvm::errs() << "Cannot convert type to bool\n";
  return nullptr;
}

} // namespace cmlirc
