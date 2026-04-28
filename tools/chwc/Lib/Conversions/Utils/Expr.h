#ifndef CHWC_UTILS_EXPR_H
#define CHWC_UTILS_EXPR_H

#include "clang/AST/Expr.h"

namespace chwc::utils {

auto ignoreCasts(clang::Expr *expr) -> clang::Expr * {
  return expr ? expr->IgnoreParenImpCasts() : nullptr;
}

} // namespace chwc::utils

#endif // CHWC_UTILS_EXPR_H
