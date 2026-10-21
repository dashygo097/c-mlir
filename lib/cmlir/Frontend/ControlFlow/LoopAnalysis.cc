#include "LoopAnalysis.h"

namespace cmlir::frontend::control {

namespace {

auto references(const clang::Stmt *stmt, const clang::VarDecl *var) -> bool {
  if (!stmt) {
    return false;
  }
  if (auto *ref = llvm::dyn_cast<clang::DeclRefExpr>(stmt)) {
    if (ref->getDecl() == var) {
      return true;
    }
  }
  for (auto *child : stmt->children()) {
    if (references(child, var)) {
      return true;
    }
  }
  return false;
}

auto mutates(const clang::Stmt *stmt, const clang::VarDecl *var) -> bool {
  if (!stmt) {
    return false;
  }
  // Calls can modify escaped bounds or induction variables.
  if (llvm::isa<clang::CallExpr>(stmt)) {
    return true;
  }
  if (auto *op = llvm::dyn_cast<clang::BinaryOperator>(stmt)) {
    if (op->isAssignmentOp()) {
      auto *lhs = op->getLHS()->IgnoreParenImpCasts();
      if (auto *ref = llvm::dyn_cast<clang::DeclRefExpr>(lhs)) {
        if (ref->getDecl() == var) {
          return true;
        }
      }
      if (auto *unary = llvm::dyn_cast<clang::UnaryOperator>(lhs)) {
        if (unary->getOpcode() == clang::UO_Deref) {
          return true;
        }
      }
    }
  }
  if (auto *op = llvm::dyn_cast<clang::UnaryOperator>(stmt)) {
    if ((op->isIncrementDecrementOp() || op->getOpcode() == clang::UO_AddrOf) &&
        references(op->getSubExpr(), var)) {
      return true;
    }
  }
  for (auto *child : stmt->children()) {
    if (mutates(child, var)) {
      return true;
    }
  }
  return false;
}

auto addressEscapes(const clang::Stmt *stmt, const clang::VarDecl *var)
    -> bool {
  if (!stmt) {
    return false;
  }
  if (auto *unary = llvm::dyn_cast<clang::UnaryOperator>(stmt)) {
    if (unary->getOpcode() == clang::UO_AddrOf &&
        references(unary->getSubExpr(), var)) {
      return true;
    }
  }
  for (auto *child : stmt->children()) {
    if (addressEscapes(child, var)) {
      return true;
    }
  }
  return false;
}

auto invariant(const clang::Stmt *expr, const clang::Stmt *body) -> bool {
  if (auto *ref = llvm::dyn_cast<clang::DeclRefExpr>(expr)) {
    if (auto *var = llvm::dyn_cast<clang::VarDecl>(ref->getDecl())) {
      if (var->getType().isVolatileQualified() || mutates(body, var) ||
          (llvm::isa<clang::FunctionDecl>(var->getDeclContext()) &&
           addressEscapes(llvm::cast<clang::FunctionDecl>(var->getDeclContext())
                              ->getBody(),
                          var))) {
        return false;
      }
    }
  }
  for (auto *child : expr->children()) {
    if (child && !invariant(child, body)) {
      return false;
    }
  }
  return true;
}

} // namespace

auto hasControlExit(const clang::Stmt *stmt) -> bool {
  if (!stmt) {
    return false;
  }
  if (llvm::isa<clang::ReturnStmt, clang::BreakStmt, clang::ContinueStmt,
                clang::SwitchStmt>(stmt)) {
    return true;
  }
  for (auto *child : stmt->children()) {
    if (hasControlExit(child)) {
      return true;
    }
  }
  return false;
}

auto analyseForLoop(clang::ForStmt *stmt, clang::ASTContext &ast)
    -> std::optional<ForLoopPlan> {
  auto *init = llvm::dyn_cast_or_null<clang::DeclStmt>(stmt->getInit());
  if (!init || !init->isSingleDecl()) {
    return {};
  }
  auto *var = llvm::dyn_cast<clang::VarDecl>(init->getSingleDecl());
  if (!var || !var->hasInit() || !var->getType()->isSignedIntegerType() ||
      var->getType().isVolatileQualified()) {
    return {};
  }
  auto *cond = llvm::dyn_cast_or_null<clang::BinaryOperator>(stmt->getCond());
  if (!cond) {
    return {};
  }
  auto *lhs =
      llvm::dyn_cast<clang::DeclRefExpr>(cond->getLHS()->IgnoreParenImpCasts());
  if (!lhs || lhs->getDecl() != var) {
    return {};
  }
  auto op = cond->getOpcode();
  bool increasing = op == clang::BO_LT || op == clang::BO_LE;
  if (!increasing && op != clang::BO_GT && op != clang::BO_GE) {
    return {};
  }
  if (var->getInit()->HasSideEffects(ast) ||
      cond->getRHS()->HasSideEffects(ast) ||
      !invariant(cond->getRHS(), stmt->getBody()) ||
      mutates(stmt->getBody(), var)) {
    return {};
  }
  int64_t step = 1;
  auto *increment = stmt->getInc();
  if (auto *unary = llvm::dyn_cast_or_null<clang::UnaryOperator>(increment)) {
    auto *ref = llvm::dyn_cast<clang::DeclRefExpr>(
        unary->getSubExpr()->IgnoreParenImpCasts());
    if (!ref || ref->getDecl() != var || !unary->isIncrementDecrementOp() ||
        unary->isIncrementOp() != increasing) {
      return {};
    }
  } else {
    auto *binary = llvm::dyn_cast_or_null<clang::BinaryOperator>(increment);
    if (!binary) {
      return {};
    }
    auto *ref = llvm::dyn_cast<clang::DeclRefExpr>(
        binary->getLHS()->IgnoreParenImpCasts());
    if (!ref || ref->getDecl() != var) {
      return {};
    }
    clang::Expr *stepExpr = nullptr;
    if (binary->getOpcode() ==
        (increasing ? clang::BO_AddAssign : clang::BO_SubAssign)) {
      stepExpr = binary->getRHS();
    } else if (binary->getOpcode() == clang::BO_Assign) {
      auto *rhs = llvm::dyn_cast<clang::BinaryOperator>(
          binary->getRHS()->IgnoreParenImpCasts());
      if (!rhs ||
          rhs->getOpcode() != (increasing ? clang::BO_Add : clang::BO_Sub)) {
        return {};
      }
      auto *self = llvm::dyn_cast<clang::DeclRefExpr>(
          rhs->getLHS()->IgnoreParenImpCasts());
      if (!self || self->getDecl() != var) {
        return {};
      }
      stepExpr = rhs->getRHS();
    }
    clang::Expr::EvalResult result;
    if (!stepExpr || stepExpr->HasSideEffects(ast) ||
        !stepExpr->EvaluateAsInt(result, ast)) {
      return {};
    }
    auto integer = result.Val.getInt();
    if (!integer.isSignedIntN(64)) {
      return {};
    }
    step = integer.getSExtValue();
    if (step <= 0) {
      return {};
    }
  }
  return ForLoopPlan{var, cond->getRHS(), step, op, increasing};
}

} // namespace cmlir::frontend::control
