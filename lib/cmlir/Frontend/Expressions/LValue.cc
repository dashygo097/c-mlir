#include "ExpressionLowering.h"

namespace cmlir::frontend {

auto ExpressionLowering::emitLValue(clang::Expr *expr, mlir::Type resultType,
                                    LValueConsumer consume)
    -> mlir::FailureOr<mlir::Value> {
  expr = expr->IgnoreParens();
  if (auto *cleanup = llvm::dyn_cast<clang::ExprWithCleanups>(expr)) {
    return emitLValue(cleanup->getSubExpr(), resultType, consume);
  }
  if (auto *argument = llvm::dyn_cast<clang::CXXDefaultArgExpr>(expr)) {
    return emitLValue(argument->getExpr(), resultType, consume);
  }
  if (auto *cast = llvm::dyn_cast<clang::CastExpr>(expr)) {
    if (cast->getCastKind() == clang::CK_NoOp) {
      return emitLValue(cast->getSubExpr(), resultType, consume);
    }
  }
  if (auto *conditional = llvm::dyn_cast<clang::ConditionalOperator>(expr)) {
    return emitConditionalLValue(*conditional, resultType, consume);
  }
  if (auto *assignment = llvm::dyn_cast<clang::BinaryOperator>(expr)) {
    if (assignment->isAssignmentOp()) {
      return emitAssignmentLValue(*assignment, resultType, consume);
    }
  }
  if (auto *unary = llvm::dyn_cast<clang::UnaryOperator>(expr)) {
    if (unary->isIncrementDecrementOp()) {
      return emitIncDecLValue(*unary, resultType, consume);
    }
  }
  if (auto *reference = llvm::dyn_cast<clang::DeclRefExpr>(expr)) {
    if (auto *parameter =
            llvm::dyn_cast<clang::ParmVarDecl>(reference->getDecl())) {
      if (auto value = function.parameters.lookup(parameter)) {
        return consume(value);
      }
    }
  }
  auto address = emitStorageAddress(expr);
  if (mlir::failed(address)) {
    return mlir::failure();
  }
  return consume(ObjectResult{*address, {}});
}

} // namespace cmlir::frontend
