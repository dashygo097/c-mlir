#ifndef CHWC_UTILS_EXPR_H
#define CHWC_UTILS_EXPR_H

#include "clang/AST/Expr.h"
#include "clang/AST/ExprCXX.h"
#include "llvm/Support/Casting.h"

namespace chwc::utils {

inline auto ignoreCasts(clang::Expr *expr) -> clang::Expr * {
  if (!expr) {
    return nullptr;
  }

  return expr->IgnoreParenImpCasts();
}

inline auto ignoreExprWrappers(clang::Expr *expr) -> clang::Expr * {
  while (expr) {
    expr = expr->IgnoreParenImpCasts();

    if (auto *cleanups =
            llvm::dyn_cast_or_null<clang::ExprWithCleanups>(expr)) {
      expr = cleanups->getSubExpr();
      continue;
    }

    if (auto *bind =
            llvm::dyn_cast_or_null<clang::CXXBindTemporaryExpr>(expr)) {
      expr = bind->getSubExpr();
      continue;
    }

    if (auto *materialize =
            llvm::dyn_cast_or_null<clang::MaterializeTemporaryExpr>(expr)) {
      expr = materialize->getSubExpr();
      continue;
    }

    return expr;
  }

  return nullptr;
}

inline auto getFieldDeclFromExpr(clang::Expr *expr)
    -> const clang::FieldDecl * {
  expr = ignoreExprWrappers(expr);

  if (auto *memberExpr = llvm::dyn_cast_or_null<clang::MemberExpr>(expr)) {
    return llvm::dyn_cast<clang::FieldDecl>(memberExpr->getMemberDecl());
  }

  if (auto *declRef = llvm::dyn_cast_or_null<clang::DeclRefExpr>(expr)) {
    return llvm::dyn_cast<clang::FieldDecl>(declRef->getDecl());
  }

  return nullptr;
}

inline auto getArrayBaseFieldDecl(clang::Expr *expr)
    -> const clang::FieldDecl * {
  expr = ignoreExprWrappers(expr);

  if (const clang::FieldDecl *fieldDecl = getFieldDeclFromExpr(expr)) {
    return fieldDecl;
  }

  if (auto *unary = llvm::dyn_cast_or_null<clang::UnaryOperator>(expr)) {
    if (unary->getOpcode() == clang::UO_Deref) {
      return getArrayBaseFieldDecl(unary->getSubExpr());
    }
  }

  return nullptr;
}

} // namespace chwc::utils

#endif // CHWC_UTILS_EXPR_H
