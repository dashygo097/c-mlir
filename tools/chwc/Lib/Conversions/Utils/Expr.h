#ifndef CHWC_UTILS_EXPR_H
#define CHWC_UTILS_EXPR_H

#include "clang/AST/Expr.h"
#include "llvm/Support/Casting.h"

namespace chwc::utils {

inline auto ignoreCasts(clang::Expr *expr) -> clang::Expr * {
  if (!expr) {
    return nullptr;
  }

  return expr->IgnoreParenImpCasts();
}

inline auto getFieldDeclFromExpr(clang::Expr *expr)
    -> const clang::FieldDecl * {
  expr = ignoreCasts(expr);

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
  expr = ignoreCasts(expr);

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
