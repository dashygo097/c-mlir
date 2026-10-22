#ifndef CLANG_MLIR_FLOATING_POINT_H
#define CLANG_MLIR_FLOATING_POINT_H

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "clang/AST/Expr.h"

namespace clang_mlir {

// Read expression-local overrides as well as command-line floating-point
// policy.
auto getFastMathFlags(const clang::Expr &expr, const clang::LangOptions &lang)
    -> mlir::arith::FastMathFlags;

} // namespace clang_mlir

#endif // CLANG_MLIR_FLOATING_POINT_H
