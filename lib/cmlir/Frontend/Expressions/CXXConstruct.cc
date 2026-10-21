#include "../Lowering.h"
#include "clang-mlir/Constants.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "llvm/Support/WithColor.h"

namespace cmlir::frontend {

auto ASTLowering::generateCXXConstructExpr(
    clang::CXXConstructExpr *constructExpr) -> mlir::Value {
  mlir::OpBuilder &builder = context.builder();
  mlir::Location loc = builder.getUnknownLoc();

  clang::QualType type = constructExpr->getType();

  if (type->isStructureOrClassType()) {
    mlir::Type structType = types.convertType(type);
    auto llvmStructType =
        mlir::dyn_cast<mlir::LLVM::LLVMStructType>(structType);

    if (!llvmStructType) {
      llvm::WithColor::error()
          << "clang-to-mlir: expected LLVM struct type for constructor\n";
      return nullptr;
    }

    clang::CXXConstructorDecl *ctor = constructExpr->getConstructor();
    if (!ctor || !ctor->isTrivial()) {
      context.source().error(constructExpr->getExprLoc(),
                             "nontrivial constructors are unsupported");
      return {};
    }
    if (ctor && ctor->isCopyOrMoveConstructor()) {
      if (constructExpr->getNumArgs() == 1) {
        auto *argument = constructExpr->getArg(0);
        return argument->isGLValue() ? loadLValue(argument)
                                     : generateExpr(argument);
      }
    }

    if (constructExpr->getNumArgs() == 0) {
      if (constructExpr->requiresZeroInitialization()) {
        return mlir::LLVM::ZeroOp::create(builder, loc, llvmStructType);
      }
      return mlir::LLVM::UndefOp::create(builder, loc, llvmStructType);
    }

    mlir::Value result =
        mlir::LLVM::UndefOp::create(builder, loc, llvmStructType).getResult();

    for (uint32_t i = 0; i < constructExpr->getNumArgs(); ++i) {
      clang::Expr *arg = constructExpr->getArg(i);
      mlir::Value argValue = generateExpr(arg);

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
    mlir::Type mlirType = types.convertType(type);

    if (auto intType = mlir::dyn_cast<mlir::IntegerType>(mlirType)) {
      return clang_mlir::intConst(builder, loc, intType, 0);
    } else if (auto floatType = mlir::dyn_cast<mlir::FloatType>(mlirType)) {
      return clang_mlir::floatConst(builder, loc, floatType, 0.0);
    }
  }

  llvm::WithColor::error()
      << "clang-to-mlir: unsupported constructor expression\n";
  return nullptr;
}

} // namespace cmlir::frontend
