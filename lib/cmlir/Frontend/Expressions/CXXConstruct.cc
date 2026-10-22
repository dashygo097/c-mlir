#include "../Lowering.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"

namespace cmlir::frontend {

auto ASTLowering::generateCXXConstructExpr(clang::CXXConstructExpr *expr)
    -> mlir::Value {
  auto *constructor = expr->getConstructor();
  if (!constructor->isTrivial()) {
    context.source().error(expr->getExprLoc(),
                           "nontrivial constructors are unsupported");
    return {};
  }
  if (constructor->isCopyOrMoveConstructor() && expr->getNumArgs() == 1) {
    auto *argument = expr->getArg(0);
    return argument->isGLValue() ? loadLValue(argument)
                                 : generateExpr(argument);
  }
  auto type = types.convertStorageType(expr->getType());
  if (!type) {
    return {};
  }
  auto &builder = context.builder();
  auto loc = context.source().location(expr->getExprLoc());
  if (expr->getNumArgs() == 0) {
    return expr->requiresZeroInitialization()
               ? mlir::LLVM::ZeroOp::create(builder, loc, type).getResult()
               : mlir::LLVM::UndefOp::create(builder, loc, type).getResult();
  }
  context.source().error(expr->getExprLoc(),
                         "unsupported constructor arguments");
  return {};
}

} // namespace cmlir::frontend
