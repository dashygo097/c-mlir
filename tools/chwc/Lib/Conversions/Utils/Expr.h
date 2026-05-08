#ifndef CHWC_UTILS_EXPR_H
#define CHWC_UTILS_EXPR_H

#include "clang/AST/Expr.h"

namespace chwc::utils {

inline auto ignoreCasts(clang::Expr *expr) -> clang::Expr * {
  if (!expr) {
    return nullptr;
  }

  return expr->IgnoreParenImpCasts();
}

} // namespace chwc::utils

#endif // CHWC_UTILS_EXPR_H
