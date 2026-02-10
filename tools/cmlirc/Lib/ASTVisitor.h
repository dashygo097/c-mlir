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
  bool TraverseVarDecl(clang::VarDecl *decl);
  bool TraverseStmt(clang::Stmt *stmt);

  // return
  bool TraverseReturnStmt(clang::ReturnStmt *retStmt);

  // control flow
  bool TraverseIfStmt(clang::IfStmt *ifStmt);
  // bool TraverseWhileStmt(clang::WhileStmt *whileStmt);
  // bool TraverseDoStmt(clang::DoStmt *doStmt);
  bool TraverseForStmt(clang::ForStmt *forStmt);
  // bool TraverseBreakStmt(clang::BreakStmt *breakStmt);
  // bool TraverseContinueStmt(clang::ContinueStmt *continueStmt);

  ContextManager &context_manager_;

private:
  // states
  llvm::DenseMap<const clang::VarDecl *, mlir::Value> symbolTable;
  llvm::DenseMap<const clang::ParmVarDecl *, mlir::Value> paramTable;
  llvm::DenseMap<const clang::FunctionDecl *, mlir::Value> functionTable;
  mlir::func::FuncOp currentFunc;

  struct ArrayAccessInfo {
    mlir::Value base;
    llvm::SmallVector<mlir::Value, 4> indices;
  };
  std::optional<ArrayAccessInfo> lastArrayAccess_;

  struct LoopContext {
    mlir::Block *headerBlock;
    mlir::Block *exitBlock;
  };
  llvm::SmallVector<LoopContext, 4> loopStack_;

  // helpers
  [[nodiscard]] bool hasSideEffects(clang::Expr *expr) const;

  // type traits
  mlir::Value convertToBool(mlir::Value value);

  // expr traits
  mlir::Value generateExpr(clang::Expr *expr, bool needLValue = false);

  mlir::Value generateBoolLiteral(clang::CXXBoolLiteralExpr *boolLit);
  mlir::Value generateIntegerLiteral(clang::IntegerLiteral *intLit);
  mlir::Value generateFloatingLiteral(clang::FloatingLiteral *floatLit);
  mlir::Value generateDeclRefExpr(clang::DeclRefExpr *declRef, bool needLValue);
  mlir::Value generateArraySubscriptExpr(clang::ArraySubscriptExpr *expr,
                                         bool needLValue);

  // unary
  mlir::Value generateIncrementDecrement(clang::Expr *expr, bool isIncrement,
                                         bool isPrefix);
  mlir::Value generateUnaryOperator(clang::UnaryOperator *unOp);

  // binary
  mlir::Value generateBinaryOperator(clang::BinaryOperator *binOp);

  // call
  mlir::Value generateCallExpr(clang::CallExpr *callExpr);
};

} // namespace cmlirc

#endif // CMLIRC_ASTVISITOR_H
