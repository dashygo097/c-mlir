#ifndef CMLIRC_ASTVISITOR_H
#define CMLIRC_ASTVISITOR_H

#include "./ContextManager.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "clang/AST/RecursiveASTVisitor.h"

namespace cmlirc {

class CMLIRCASTVisitor : public clang::RecursiveASTVisitor<CMLIRCASTVisitor> {
public:
  explicit CMLIRCASTVisitor(ContextManager &ctx);
  ~CMLIRCASTVisitor() = default;

  bool TraverseFunctionDecl(clang::FunctionDecl *D);
  bool TraverseStmt(clang::Stmt *stmt);
  bool TraverseVarDecl(clang::VarDecl *decl);
  bool TraverseReturnStmt(clang::ReturnStmt *retStmt);

  ContextManager &context_manager_;

private:
  // states
  llvm::DenseMap<const clang::VarDecl *, mlir::Value> symbolTable;
  llvm::DenseMap<const clang::ParmVarDecl *, mlir::Value> paramTable;
  llvm::DenseMap<const clang::FunctionDecl *, mlir::Value> functionTable;
  mlir::func::FuncOp currentFunc;

  // helpers
  [[nodiscard]] bool hasSideEffects(clang::Expr *expr) const;

  // type traits

  // expr traits
  mlir::Value generateExpr(clang::Expr *expr, bool needLValue = false);

  mlir::Value generateIntegerLiteral(clang::IntegerLiteral *intLit);
  mlir::Value generateFloatingLiteral(clang::FloatingLiteral *floatLit);
  mlir::Value generateDeclRefExpr(clang::DeclRefExpr *declRef, bool needLValue);
  mlir::Value generateArraySubscriptExpr(clang::ArraySubscriptExpr *expr,
                                         bool needLValue);
  mlir::Value generateIncrementDecrement(clang::Expr *expr, bool isIncrement,
                                         bool isPrefix);

  mlir::Value generateUnaryOperator(clang::UnaryOperator *unOp);
  mlir::Value generateBinaryOperator(clang::BinaryOperator *binOp);

  mlir::Value generateCallExpr(clang::CallExpr *callExpr);
};

} // namespace cmlirc

#endif // CMLIRC_ASTVISITOR_H
