#include "../../Converter.h"
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

  REGISTER_EXPR_CONVERSION(ExprWithCleanups)
  REGISTER_EXPR_CONVERSION(CXXBindTemporaryExpr)
  REGISTER_EXPR_CONVERSION(MaterializeTemporaryExpr)
  REGISTER_EXPR_CONVERSION(CXXConstructExpr)
  REGISTER_EXPR_CONVERSION(CXXBoolLiteralExpr)
  REGISTER_EXPR_CONVERSION(IntegerLiteral)
  REGISTER_EXPR_CONVERSION(DeclRefExpr)
  REGISTER_EXPR_CONVERSION(ImplicitCastExpr)
  REGISTER_EXPR_CONVERSION(MemberExpr)
  REGISTER_EXPR_CONVERSION(CXXMemberCallExpr)
  REGISTER_EXPR_CONVERSION(CXXOperatorCallExpr)
  REGISTER_EXPR_CONVERSION(CXXFunctionalCastExpr)
  REGISTER_EXPR_CONVERSION(UnaryOperator)
  REGISTER_EXPR_CONVERSION(BinaryOperator)

#undef REGISTER_EXPR_CONVERSION

  llvm::WithColor::error()
      << "chwc: unsupported expression conversion for expr: "
      << expr->getStmtClassName() << "\n";
  return nullptr;
}

} // namespace chwc
