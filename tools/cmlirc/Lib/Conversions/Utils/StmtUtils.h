#ifndef CMLIRC_STMTUTILS_H
#define CMLIRC_STMTUTILS_H

#include "clang/AST/Stmt.h"

namespace cmlirc::detail {

inline bool stmtHasReturnRecursively(const clang::Stmt *stmt) {
  if (!stmt)
    return false;
  if (llvm::isa<clang::ReturnStmt>(stmt))
    return true;
  for (const auto *child : stmt->children())
    if (stmtHasReturnRecursively(child))
      return true;
  return false;
}

inline bool stmtHasReturnInIf(const clang::Stmt *stmt) {
  if (!stmt)
    return false;
  for (const auto *child : stmt->children()) {
    if (const auto *ifStmt = llvm::dyn_cast_or_null<clang::IfStmt>(child)) {
      if (stmtHasReturnRecursively(ifStmt->getThen()) ||
          stmtHasReturnRecursively(ifStmt->getElse()))
        return true;
    }
  }
  return false;
}

inline bool stmtHasReturnInLoop(const clang::Stmt *stmt) {
  if (!stmt)
    return false;
  for (const auto *child : stmt->children())
    if (llvm::isa<clang::ForStmt, clang::WhileStmt, clang::DoStmt>(child))
      if (stmtHasReturnRecursively(child))
        return true;
  return false;
}

inline bool stmtHasReturnInSwitch(const clang::Stmt *stmt) {
  if (!stmt)
    return false;
  for (const auto *child : stmt->children())
    if (llvm::isa<clang::SwitchStmt>(child))
      if (stmtHasReturnRecursively(child))
        return true;
  return false;
}

inline bool stmtHasReturnInAnyControlFlow(const clang::Stmt *stmt) {
  return stmtHasReturnInIf(stmt) || stmtHasReturnInLoop(stmt) ||
         stmtHasReturnInSwitch(stmt);
}

inline bool stmtHasBreakRecursively(const clang::Stmt *stmt) {
  if (!stmt)
    return false;
  if (llvm::isa<clang::BreakStmt>(stmt))
    return true;
  if (llvm::isa<clang::ForStmt, clang::WhileStmt, clang::DoStmt,
                clang::SwitchStmt>(stmt))
    return false;
  for (const auto *child : stmt->children())
    if (stmtHasBreakRecursively(child))
      return true;
  return false;
}

inline bool stmtHasBreakInLoop(const clang::Stmt *forOrWhileOrDo) {
  if (!forOrWhileOrDo)
    return false;
  const clang::Stmt *body = nullptr;
  if (const auto *f = llvm::dyn_cast<clang::ForStmt>(forOrWhileOrDo))
    body = f->getBody();
  else if (const auto *w = llvm::dyn_cast<clang::WhileStmt>(forOrWhileOrDo))
    body = w->getBody();
  else if (const auto *d = llvm::dyn_cast<clang::DoStmt>(forOrWhileOrDo))
    body = d->getBody();
  else
    return false;
  return stmtHasBreakRecursively(body);
}

inline bool stmtHasContinueRecursively(const clang::Stmt *stmt) {
  if (!stmt)
    return false;
  if (llvm::isa<clang::ContinueStmt>(stmt))
    return true;
  if (llvm::isa<clang::ForStmt, clang::WhileStmt, clang::DoStmt>(stmt))
    return false;
  for (const auto *child : stmt->children())
    if (stmtHasContinueRecursively(child))
      return true;
  return false;
}

inline bool stmtHasContinueInLoop(const clang::Stmt *forOrWhileOrDo) {
  if (!forOrWhileOrDo)
    return false;
  const clang::Stmt *body = nullptr;
  if (const auto *f = llvm::dyn_cast<clang::ForStmt>(forOrWhileOrDo))
    body = f->getBody();
  else if (const auto *w = llvm::dyn_cast<clang::WhileStmt>(forOrWhileOrDo))
    body = w->getBody();
  else if (const auto *d = llvm::dyn_cast<clang::DoStmt>(forOrWhileOrDo))
    body = d->getBody();
  else
    return false;
  return stmtHasContinueRecursively(body);
}

} // namespace cmlirc::detail

#endif // CMLIRC_STMTUTILS_H
