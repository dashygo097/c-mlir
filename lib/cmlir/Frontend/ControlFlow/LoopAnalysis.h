#ifndef CMLIR_FRONTEND_LOOP_ANALYSIS_H
#define CMLIR_FRONTEND_LOOP_ANALYSIS_H

#include "clang/AST/ASTContext.h"
#include "clang/AST/Expr.h"
#include "clang/AST/Stmt.h"
#include <optional>

namespace cmlir::frontend::control {

struct ForLoopPlan {
  clang::VarDecl *induction;
  clang::Expr *bound;
  int64_t step;
  clang::BinaryOperatorKind comparison;
  bool increasing;
};

// Recognition never emits IR or evaluates source expressions at runtime.
auto analyseForLoop(clang::ForStmt *, clang::ASTContext &)
    -> std::optional<ForLoopPlan>;

// Conservatively identify loop bodies that require CFG lowering.
auto hasControlExit(const clang::Stmt *stmt) -> bool;

} // namespace cmlir::frontend::control

#endif // CMLIR_FRONTEND_LOOP_ANALYSIS_H
