#include "../../Converter.h"

namespace cmlirc {

mlir::Value CMLIRConverter::generateExpr(clang::Expr *expr) {
  if (!expr)
    return nullptr;

#define REGISTER_EXPR_CONVERSION(ExprType)                                     \
  if (auto *node = mlir::dyn_cast<clang::ExprType>(expr)) {                    \
    return generate##ExprType(node);                                           \
  }

  REGISTER_EXPR_CONVERSION(ParenExpr)
  REGISTER_EXPR_CONVERSION(CXXBoolLiteralExpr)
  REGISTER_EXPR_CONVERSION(IntegerLiteral)
  REGISTER_EXPR_CONVERSION(FloatingLiteral)
  REGISTER_EXPR_CONVERSION(DeclRefExpr)
  REGISTER_EXPR_CONVERSION(ImplicitCastExpr)
  REGISTER_EXPR_CONVERSION(ArraySubscriptExpr)
  REGISTER_EXPR_CONVERSION(UnaryOperator)
  REGISTER_EXPR_CONVERSION(BinaryOperator)
  REGISTER_EXPR_CONVERSION(ConditionalOperator)
  REGISTER_EXPR_CONVERSION(InitListExpr)
  REGISTER_EXPR_CONVERSION(CallExpr)
  REGISTER_EXPR_CONVERSION(CXXConstructExpr)
  REGISTER_EXPR_CONVERSION(MemberExpr)

#undef REGISTER_EXPR_CONVERSION

  llvm::errs() << "Unsupported expression conversion for expr: "
               << expr->getStmtClassName() << "\n";
  return nullptr;
}

} // namespace cmlirc
