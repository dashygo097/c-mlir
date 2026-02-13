#ifndef CMLIRC_ASTVISITOR_H
#define CMLIRC_ASTVISITOR_H

#include "./Context/ContextManager.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "clang/AST/RecursiveASTVisitor.h"

namespace cmlirc {

class CMLIRConverter : public clang::RecursiveASTVisitor<CMLIRConverter> {
public:
  explicit CMLIRConverter(ContextManager &ctx) : context_manager_(ctx) {}
  ~CMLIRConverter() = default;

  // decl traits
  bool TraverseFunctionDecl(clang::FunctionDecl *D);
  bool TraverseVarDecl(clang::VarDecl *decl);

  // stmt traits
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

private:
  ContextManager &context_manager_;

  // states
  llvm::DenseMap<const clang::VarDecl *, mlir::Value> symbolTable;
  llvm::DenseMap<const clang::ParmVarDecl *, mlir::Value> paramTable;
  llvm::DenseMap<const clang::FunctionDecl *, mlir::Value> functionTable;
  mlir::func::FuncOp currentFunc;
  mlir::Value *returnValueCapture;

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
  bool branchEndsWithReturn(clang::Stmt *stmt);

  // type traits

  // expr traits
  mlir::Value generateExpr(clang::Expr *expr);

  // paren
  mlir::Value generateParenExpr(clang::ParenExpr *parenExpr);

  // literals
  mlir::Value generateCXXBoolLiteralExpr(clang::CXXBoolLiteralExpr *boolLit);
  mlir::Value generateIntegerLiteral(clang::IntegerLiteral *intLit);
  mlir::Value generateFloatingLiteral(clang::FloatingLiteral *floatLit);

  // decl ref
  mlir::Value generateDeclRefExpr(clang::DeclRefExpr *declRef);

  // implicit cast
  mlir::Value generateImplicitCastExpr(clang::ImplicitCastExpr *castExpr);

  // array subscript
  mlir::Value generateArraySubscriptExpr(clang::ArraySubscriptExpr *expr);

  // unary
  mlir::Value generateUnaryOperator(clang::UnaryOperator *unOp);
  mlir::Value generateIncrementDecrement(clang::Expr *expr, bool isIncrement,
                                         bool isPrefix);

  // binary
  mlir::Value generateBinaryOperator(clang::BinaryOperator *binOp);

  // call
  mlir::Value generateCallExpr(clang::CallExpr *callExpr);
};

} // namespace cmlirc

#endif // CMLIRC_ASTVISITOR_H
