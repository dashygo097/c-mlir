#include "../../Converter.h"
#include "clang/AST/Expr.h"
#include "llvm/Support/WithColor.h"

namespace chwc {

auto CHWConverter::generateExpr(clang::Expr *expr) -> mlir::Value {
  if (!expr) {
    return nullptr;
  }

  expr = expr->IgnoreParens();

#define REGISTER_EXPR_CONVERSION(ExprType)                                     \
  if (auto *node = mlir::dyn_cast<clang::ExprType>(expr)) {                    \
    return generate##ExprType(node);                                           \
  }

  REGISTER_EXPR_CONVERSION(CXXBoolLiteralExpr)
  REGISTER_EXPR_CONVERSION(IntegerLiteral)
  REGISTER_EXPR_CONVERSION(DeclRefExpr)
  REGISTER_EXPR_CONVERSION(ImplicitCastExpr)
  REGISTER_EXPR_CONVERSION(MemberExpr)
  REGISTER_EXPR_CONVERSION(CXXMemberCallExpr)
  REGISTER_EXPR_CONVERSION(BinaryOperator)
  REGISTER_EXPR_CONVERSION(UnaryOperator)

#undef REGISTER_EXPR_CONVERSION

  llvm::WithColor::error()
      << "chwc: unsupported expression conversion for expr: "
      << expr->getStmtClassName() << "\n";
  return nullptr;
}

} // namespace chwc
