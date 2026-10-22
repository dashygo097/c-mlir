#include "../Lowering.h"
#include "clang/AST/Expr.h"
#include "llvm/Support/WithColor.h"

namespace cmlir::frontend {

auto ASTLowering::generateExpr(clang::Expr *expr) -> mlir::Value {
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
  REGISTER_EXPR_CONVERSION(ImplicitCastExpr)
  REGISTER_EXPR_CONVERSION(CStyleCastExpr)
  REGISTER_EXPR_CONVERSION(ArraySubscriptExpr)
  REGISTER_EXPR_CONVERSION(UnaryOperator)
  REGISTER_EXPR_CONVERSION(BinaryOperator)
  REGISTER_EXPR_CONVERSION(ConditionalOperator)
  REGISTER_EXPR_CONVERSION(InitListExpr)
  REGISTER_EXPR_CONVERSION(CallExpr)
  REGISTER_EXPR_CONVERSION(CXXConstructExpr)
  REGISTER_EXPR_CONVERSION(MemberExpr)
  REGISTER_EXPR_CONVERSION(UnaryExprOrTypeTraitExpr)
  REGISTER_EXPR_CONVERSION(CXXThisExpr)

#undef REGISTER_EXPR_CONVERSION

  context.source().error(expr->getExprLoc(), "unsupported expression");
  return nullptr;
}

} // namespace cmlir::frontend
