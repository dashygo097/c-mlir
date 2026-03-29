#ifndef CMLIRC_STMTUTILS_H
#define CMLIRC_STMTUTILS_H

#include "clang/AST/Stmt.h"

namespace cmlirc::detail {

inline auto stmtHasBreakRecursively(const clang::Stmt *stmt) -> bool {
  if (!stmt) {
    return false;
}
  if (llvm::isa<clang::BreakStmt>(stmt)) {
    return true;
}
  if (llvm::isa<clang::ForStmt, clang::WhileStmt, clang::DoStmt,
                clang::SwitchStmt>(stmt)) {
    return false;
}
  for (const auto *child : stmt->children()) {
    if (stmtHasBreakRecursively(child)) {
      return true;
}
}
  return false;
}

inline auto stmtHasBreakInLoop(const clang::Stmt *loopStmt) -> bool {
  if (!loopStmt) {
    return false;
}
  const clang::Stmt *body = nullptr;
  if (const auto *f = llvm::dyn_cast<clang::ForStmt>(loopStmt)) {
    body = f->getBody();
  } else if (const auto *w = llvm::dyn_cast<clang::WhileStmt>(loopStmt)) {
    body = w->getBody();
  } else if (const auto *d = llvm::dyn_cast<clang::DoStmt>(loopStmt)) {
    body = d->getBody();
  } else {
    return false;
}
  return stmtHasBreakRecursively(body);
}

inline auto stmtHasContinueRecursively(const clang::Stmt *stmt) -> bool {
  if (!stmt) {
    return false;
}
  if (llvm::isa<clang::ContinueStmt>(stmt)) {
    return true;
}
  if (llvm::isa<clang::ForStmt, clang::WhileStmt, clang::DoStmt>(stmt)) {
    return false;
}
  for (const auto *child : stmt->children()) {
    if (stmtHasContinueRecursively(child)) {
      return true;
}
}
  return false;
}

inline auto stmtHasContinueInLoop(const clang::Stmt *loopStmt) -> bool {
  if (!loopStmt) {
    return false;
}
  const clang::Stmt *body = nullptr;
  if (const auto *f = llvm::dyn_cast<clang::ForStmt>(loopStmt)) {
    body = f->getBody();
  } else if (const auto *w = llvm::dyn_cast<clang::WhileStmt>(loopStmt)) {
    body = w->getBody();
  } else if (const auto *d = llvm::dyn_cast<clang::DoStmt>(loopStmt)) {
    body = d->getBody();
  } else {
    return false;
}
  return stmtHasContinueRecursively(body);
}

inline auto stmtHasReturnRecursively(const clang::Stmt *stmt) -> bool {
  if (!stmt) {
    return false;
}
  if (llvm::isa<clang::ReturnStmt>(stmt)) {
    return true;
}
  for (const auto *child : stmt->children()) {
    if (stmtHasReturnRecursively(child)) {
      return true;
}
}
  return false;
}

inline auto stmtHasReturnInLoop(const clang::Stmt *loopStmt) -> bool {
  if (!loopStmt) {
    return false;
}
  const clang::Stmt *body = nullptr;
  if (const auto *f = llvm::dyn_cast<clang::ForStmt>(loopStmt)) {
    body = f->getBody();
  } else if (const auto *w = llvm::dyn_cast<clang::WhileStmt>(loopStmt)) {
    body = w->getBody();
  } else if (const auto *d = llvm::dyn_cast<clang::DoStmt>(loopStmt)) {
    body = d->getBody();
  } else {
    return false;
}
  return stmtHasReturnRecursively(body);
}

} // namespace cmlirc::detail

#endif // CMLIRC_STMTUTILS_H
