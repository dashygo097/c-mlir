#include "../../Converter.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"

namespace cmlirc {

mlir::Value CMLIRConverter::generateCXXConstructExpr(
    clang::CXXConstructExpr *constructExpr) {
  mlir::OpBuilder &builder = context_manager_.Builder();
  mlir::Location loc = builder.getUnknownLoc();

  clang::QualType type = constructExpr->getType();

  if (type->isStructureOrClassType()) {
    mlir::Type structType = convertType(type);
    auto llvmStructType =
        mlir::dyn_cast<mlir::LLVM::LLVMStructType>(structType);

    if (!llvmStructType) {
      llvm::errs() << "Expected LLVM struct type for constructor\n";
      return nullptr;
    }

    if (constructExpr->getNumArgs() == 0) {
      auto undefValue =
          mlir::LLVM::UndefOp::create(builder, loc, llvmStructType);

      mlir::Value result = undefValue.getResult();
      uint32_t numFields = llvmStructType.getBody().size();

      for (uint32_t i = 0; i < numFields; ++i) {
        mlir::Type fieldType = llvmStructType.getBody()[i];
        mlir::Value zeroValue;

        if (auto intType = mlir::dyn_cast<mlir::IntegerType>(fieldType)) {
          zeroValue =
              mlir::arith::ConstantOp::create(
                  builder, loc, intType, builder.getIntegerAttr(intType, 0))
                  .getResult();
        } else if (auto floatType =
                       mlir::dyn_cast<mlir::FloatType>(fieldType)) {
          zeroValue =
              mlir::arith::ConstantOp::create(
                  builder, loc, floatType, builder.getFloatAttr(floatType, 0.0))
                  .getResult();
        } else {
          continue;
        }

        result = mlir::LLVM::InsertValueOp::create(builder, loc, result,
                                                   zeroValue, i)
                     .getResult();
      }

      return result;
    }

    mlir::Value result =
        mlir::LLVM::UndefOp::create(builder, loc, llvmStructType).getResult();

    for (uint32_t i = 0; i < constructExpr->getNumArgs(); ++i) {
      clang::Expr *arg = constructExpr->getArg(i);
      mlir::Value argValue = generateExpr(arg);

      if (!argValue) {
        llvm::errs() << "Failed to generate constructor argument " << i << "\n";
        return nullptr;
      }

      result =
          mlir::LLVM::InsertValueOp::create(builder, loc, result, argValue, i)
              .getResult();
    }

    return result;
  }

  if (constructExpr->getNumArgs() == 1) {
    return generateExpr(constructExpr->getArg(0));
  }

  if (constructExpr->getNumArgs() == 0) {
    mlir::Type mlirType = convertType(type);

    if (auto intType = mlir::dyn_cast<mlir::IntegerType>(mlirType)) {
      return mlir::arith::ConstantOp::create(builder, loc, intType,
                                             builder.getIntegerAttr(intType, 0))
          .getResult();
    } else if (auto floatType = mlir::dyn_cast<mlir::FloatType>(mlirType)) {
      return mlir::arith::ConstantOp::create(
                 builder, loc, floatType, builder.getFloatAttr(floatType, 0.0))
          .getResult();
    }
  }

  llvm::errs() << "Unsupported constructor expression\n";
  return nullptr;
}

} // namespace cmlirc
