#include "../../Converter.h"
#include "../Utils/Constants.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "llvm/Support/WithColor.h"

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
      llvm::WithColor::error()
          << "cmlirc: expected LLVM struct type for constructor\n";
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
          zeroValue = detail::intConst(builder, loc, intType, 0);
        } else if (auto floatType =
                       mlir::dyn_cast<mlir::FloatType>(fieldType)) {
          zeroValue = detail::floatConst(builder, loc, floatType, 0.0);
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
        llvm::WithColor::error()
            << "cmlirc: failed to generate constructor argument " << i << "\n";
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

    if (auto intType = mlir::dyn_cast<mlir::IntegerType>(mlirType))
      return detail::intConst(builder, loc, intType, 0);
    else if (auto floatType = mlir::dyn_cast<mlir::FloatType>(mlirType))
      return detail::floatConst(builder, loc, floatType, 0.0);
  }

  llvm::WithColor::error() << "cmlirc: unsupported constructor expression\n";
  return nullptr;
}

} // namespace cmlirc
