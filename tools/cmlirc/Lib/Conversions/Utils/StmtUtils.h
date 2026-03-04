#ifndef CMLIRC_STMTUTILS_H
#define CMLIRC_STMTUTILS_H

#include "mlir/Support/LLVM.h"
#include "clang/AST/Stmt.h"

namespace cmlirc::detail {
inline bool stmtHasReturn(const clang::Stmt *stmt) {
  if (!stmt) {
    return false;
  }
  for (const auto *child : stmt->children()) {
    if (mlir::isa<clang::ReturnStmt>(child)) {
      return true;
    }
  }
  return false;
}

inline bool stmtHasReturnRecursively(const clang::Stmt *stmt) {
  if (!stmt) {
    return false;
  }
  if (mlir::isa<clang::ReturnStmt>(stmt)) {
    return true;
  }
  for (const auto *child : stmt->children()) {
    if (stmtHasReturnRecursively(child)) {
      return true;
    }
  }
  return false;
}

inline bool stmtHasReturnInIf(const clang::Stmt *stmt) {
  if (!stmt) {
    return false;
  }
  for (const auto *child : stmt->children()) {
    if (mlir::isa<clang::IfStmt>(child)) {
      const auto *ifStmt = mlir::cast<clang::IfStmt>(child);
      if (stmtHasReturnRecursively(ifStmt->getThen()) ||
          stmtHasReturnRecursively(ifStmt->getElse())) {
        return true;
      }
    }
  }
  return false;
}

inline bool stmtHasReturnInLoop(const clang::Stmt *stmt) {
  if (!stmt) {
    return false;
  }
  for (const auto *child : stmt->children()) {
    if (mlir::isa<clang::ForStmt>(child) ||
        mlir::isa<clang::WhileStmt>(child) || mlir::isa<clang::DoStmt>(child)) {
      if (stmtHasReturnRecursively(child)) {
        return true;
      }
    }
  }
  return false;
}

inline bool stmtHasReturnInSwitch(const clang::Stmt *stmt) {
  if (!stmt) {
    return false;
  }
  for (const auto *child : stmt->children()) {
    if (mlir::isa<clang::SwitchStmt>(child)) {
      if (stmtHasReturnRecursively(child)) {
        return true;
      }
    }
  }
  return false;
}

inline bool stmtHasReturnInAnyControlFlow(const clang::Stmt *stmt) {
  return stmtHasReturnInIf(stmt) || stmtHasReturnInLoop(stmt) ||
         stmtHasReturnInSwitch(stmt);
}

inline bool stmtHasBreak(const clang::Stmt *stmt) {
  if (!stmt) {
    return false;
  }
  for (const auto *child : stmt->children()) {
    if (mlir::isa<clang::BreakStmt>(child)) {
      return true;
    }
  }
  return false;
}

inline bool stmtHasBreakRecursively(const clang::Stmt *stmt) {
  if (!stmt) {
    return false;
  }
  if (mlir::isa<clang::BreakStmt>(stmt)) {
    return true;
  }
  for (const auto *child : stmt->children()) {
    if (stmtHasBreakRecursively(child)) {
      return true;
    }
  }
  return false;
}

inline bool stmtHasBreakInLoop(const clang::Stmt *stmt) {
  if (!stmt) {
    return false;
  }
  for (const auto *child : stmt->children()) {
    if ((mlir::isa<clang::ForStmt>(child) ||
         mlir::isa<clang::WhileStmt>(child) ||
         mlir::isa<clang::DoStmt>(child)) &&
        stmtHasBreakRecursively(child)) {
      return true;
    }
  }
  return false;
}

inline bool stmtHasContinue(const clang::Stmt *stmt) {
  if (!stmt) {
    return false;
  }
  for (const auto *child : stmt->children()) {
    if (mlir::isa<clang::ContinueStmt>(child)) {
      return true;
    }
  }
  return false;
}

inline bool stmtHasContinueRecursively(const clang::Stmt *stmt) {
  if (!stmt) {
    return false;
  }
  if (mlir::isa<clang::ContinueStmt>(stmt)) {
    return true;
  }
  for (const auto *child : stmt->children()) {
    if (stmtHasContinueRecursively(child)) {
      return true;
    }
  }
  return false;
}

inline bool stmtHasContinueInLoop(const clang::Stmt *stmt) {
  if (!stmt) {
    return false;
  }
  for (const auto *child : stmt->children()) {
    if ((mlir::isa<clang::ForStmt>(child) ||
         mlir::isa<clang::WhileStmt>(child) ||
         mlir::isa<clang::DoStmt>(child)) &&
        stmtHasContinueRecursively(child)) {
      return true;
    }
  }
  return false;
}

} // namespace cmlirc::detail

#endif // CMLIRC_STMTUTILS_H
