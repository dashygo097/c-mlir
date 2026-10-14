#include "../Lowering.h"
#include "llvm/Support/raw_ostream.h"

namespace cmlir::frontend {

auto ASTLowering::emitStatement(clang::Stmt *stmt) -> bool {
  if (!stmt) {
    return true;
  }
  if (!function.operation) {
    return false;
  }
  if (context.source().failed()) {
    return false;
  }
  auto *block = context.builder().getInsertionBlock();
  if (block && !block->empty() &&
      block->back().hasTrait<mlir::OpTrait::IsTerminator>()) {
    return true;
  }
  clang_mlir::Source::TraceScope frame(
      context.source(), stmt->getStmtClassName(), stmt->getBeginLoc(), trace);
  if (auto *expr = llvm::dyn_cast<clang::Expr>(stmt)) {
    auto value = generateExpr(expr);
    return !context.source().failed() &&
           (value || expr->getType()->isVoidType());
  }
  switch (stmt->getStmtClass()) {
  case clang::Stmt::NullStmtClass:
    return true;
  case clang::Stmt::CompoundStmtClass:
    for (auto *child : llvm::cast<clang::CompoundStmt>(stmt)->body()) {
      if (!emitStatement(child)) {
        return false;
      }
    }
    return true;
  case clang::Stmt::DeclStmtClass:
    for (auto *decl : llvm::cast<clang::DeclStmt>(stmt)->decls()) {
      if (!emitDeclaration(decl)) {
        return false;
      }
    }
    return true;
  case clang::Stmt::IfStmtClass:
    return emitIf(llvm::cast<clang::IfStmt>(stmt));
  case clang::Stmt::SwitchStmtClass:
    return emitSwitch(llvm::cast<clang::SwitchStmt>(stmt));
  case clang::Stmt::ForStmtClass:
    return emitFor(llvm::cast<clang::ForStmt>(stmt));
  case clang::Stmt::WhileStmtClass:
    return emitWhile(llvm::cast<clang::WhileStmt>(stmt));
  case clang::Stmt::DoStmtClass:
    return emitDo(llvm::cast<clang::DoStmt>(stmt));
  case clang::Stmt::ReturnStmtClass:
    return emitReturn(llvm::cast<clang::ReturnStmt>(stmt));
  case clang::Stmt::BreakStmtClass:
    return emitBreak(llvm::cast<clang::BreakStmt>(stmt));
  case clang::Stmt::ContinueStmtClass:
    return emitContinue(llvm::cast<clang::ContinueStmt>(stmt));

  default:
    context.source().error(stmt->getBeginLoc(), "unsupported statement");
    return false;
  }
}

} // namespace cmlir::frontend
