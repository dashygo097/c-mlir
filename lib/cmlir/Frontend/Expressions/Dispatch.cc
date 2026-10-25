#include "ExpressionLowering.h"
#include "clang/AST/Expr.h"
#include "llvm/Support/WithColor.h"

namespace cmlir::frontend {

auto ExpressionLowering::emitValue(clang::Expr &expr)
    -> mlir::FailureOr<mlir::Value> {
  if (expr.getType()->isVoidType()) {
    context.source().error(expr.getExprLoc(),
                           "void expression does not produce a value");
    return mlir::failure();
  }
  auto value = generateExpr(&expr);
  if (!value || context.source().failed()) {
    if (!context.source().failed()) {
      context.source().error(expr.getExprLoc(), "expression lowering failed");
    }
    return mlir::failure();
  }
  return value;
}

auto ExpressionLowering::emitAddress(clang::Expr &expr)
    -> mlir::FailureOr<clang_mlir::Address> {
  if (context.source().failed()) {
    return mlir::failure();
  }
  auto address = addressOf(&expr);
  if (mlir::failed(address) && !context.source().failed()) {
    context.source().error(expr.getExprLoc(), "object address lowering failed");
  }
  return address;
}

auto ExpressionLowering::emitDiscarded(clang::Expr &expr)
    -> mlir::LogicalResult {
  if (context.source().failed()) {
    return mlir::failure();
  }
  auto *node = expr.IgnoreParens();
  if (auto *cleanup = llvm::dyn_cast<clang::ExprWithCleanups>(node)) {
    return emitDiscarded(*cleanup->getSubExpr());
  }
  if (auto *cast = llvm::dyn_cast<clang::CastExpr>(node)) {
    if (cast->getCastKind() == clang::CK_ToVoid ||
        cast->getCastKind() == clang::CK_NoOp) {
      return emitDiscarded(*cast->getSubExpr());
    }
  }
  bool isUpdate = false;
  if (auto *binary = llvm::dyn_cast<clang::BinaryOperator>(node)) {
    isUpdate = binary->isAssignmentOp();
  }
  if (auto *unary = llvm::dyn_cast<clang::UnaryOperator>(node)) {
    isUpdate = unary->isIncrementDecrementOp();
  }
  bool constrainedFP = node->getType()->isFloatingType() &&
                       node->getFPFeaturesInEffect(context.ast().getLangOpts())
                           .isFPConstrained();
  if (node->isGLValue() && !node->getType()->isFunctionType() && !isUpdate &&
      !constrainedFP && !node->getType().isVolatileQualified()) {
    auto emitted =
        emitLValue(node, context.builder().getNoneType(),
                   [](const LValueResult &) -> mlir::FailureOr<mlir::Value> {
                     return mlir::Value{};
                   });
    if (mlir::failed(emitted) && !context.source().failed()) {
      context.source().error(expr.getExprLoc(),
                             "discarded expression lowering failed");
    }
    return mlir::success(mlir::succeeded(emitted));
  }
  auto value = generateExpr(node);
  return mlir::success(!context.source().failed() &&
                       (value || node->getType()->isVoidType()));
}

auto ExpressionLowering::generateExpr(clang::Expr *expr) -> mlir::Value {
  if (context.source().failed()) {
    return nullptr;
  }
  if (!expr) {
    return nullptr;
  }
  if (expr->getType()->isFloatingType() &&
      expr->getFPFeaturesInEffect(context.ast().getLangOpts())
          .isFPConstrained()) {
    context.source().error(
        expr->getExprLoc(),
        "constrained floating-point expressions are unsupported");
    return nullptr;
  }

  clang_mlir::Source::TraceScope frame(
      context.source(), expr->getStmtClassName(), expr->getExprLoc(), trace);

  if (auto *temporary = llvm::dyn_cast<clang::MaterializeTemporaryExpr>(expr)) {
    return generateExpr(temporary->getSubExpr());
  }
  if (auto *cleanup = llvm::dyn_cast<clang::ExprWithCleanups>(expr)) {
    return generateExpr(cleanup->getSubExpr());
  }

  if (auto *argument = llvm::dyn_cast<clang::CXXDefaultArgExpr>(expr)) {
    return generateExpr(argument->getExpr());
  }
  if (expr->isGLValue() && !expr->getType()->isArrayType() &&
      !expr->getType()->isFunctionType()) {
    return loadLValue(expr);
  }

  if (auto *call = llvm::dyn_cast<clang::CallExpr>(expr)) {
    auto result = generateCallExpr(call);
    if (mlir::failed(result)) {
      if (!context.source().failed()) {
        context.source().error(expr->getExprLoc(),
                               "function call lowering failed");
      }
      return {};
    }
    return *result;
  }

#define REGISTER_EXPR_CONVERSION(ExprType)                                     \
  if (auto *node = mlir::dyn_cast<clang::ExprType>(expr)) {                    \
    auto result = generate##ExprType(node);                                    \
    if (!result && !expr->getType()->isVoidType() &&                           \
        !context.source().failed()) {                                          \
      context.source().error(expr->getExprLoc(),                               \
                             "expression lowering failed");                    \
    }                                                                          \
    return result;                                                             \
  }

  REGISTER_EXPR_CONVERSION(ParenExpr)
  REGISTER_EXPR_CONVERSION(CXXBoolLiteralExpr)
  REGISTER_EXPR_CONVERSION(IntegerLiteral)
  REGISTER_EXPR_CONVERSION(FloatingLiteral)
  REGISTER_EXPR_CONVERSION(CharacterLiteral)
  REGISTER_EXPR_CONVERSION(StringLiteral)
  REGISTER_EXPR_CONVERSION(DeclRefExpr)
  REGISTER_EXPR_CONVERSION(CastExpr)
  REGISTER_EXPR_CONVERSION(ArraySubscriptExpr)
  REGISTER_EXPR_CONVERSION(UnaryOperator)
  REGISTER_EXPR_CONVERSION(BinaryOperator)
  REGISTER_EXPR_CONVERSION(ConditionalOperator)
  REGISTER_EXPR_CONVERSION(InitListExpr)
  REGISTER_EXPR_CONVERSION(CXXConstructExpr)
  REGISTER_EXPR_CONVERSION(MemberExpr)
  REGISTER_EXPR_CONVERSION(UnaryExprOrTypeTraitExpr)
  REGISTER_EXPR_CONVERSION(CXXThisExpr)

#undef REGISTER_EXPR_CONVERSION

  context.source().error(expr->getExprLoc(), "unsupported expression");
  return nullptr;
}

} // namespace cmlir::frontend
