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

  bool VisitVarDecl(clang::VarDecl *decl);
  bool VisitReturnStmt(clang::ReturnStmt *retStmt);

  ContextManager &context_manager_;

private:
  // states
  llvm::DenseMap<const clang::VarDecl *, mlir::Value> symbolTable;
  llvm::DenseMap<const clang::ParmVarDecl *, mlir::Value> paramTable;
  mlir::func::FuncOp currentFunc;

  // type traits

  // expr traits
  mlir::Value generateExpr(clang::Expr *expr);

  mlir::Value generateIntegerLiteral(clang::IntegerLiteral *intLit);
  mlir::Value generateFloatingLiteral(clang::FloatingLiteral *floatLit);
  mlir::Value generateDeclRefExpr(clang::DeclRefExpr *declRef);

  mlir::Value generateUnaryOperator(clang::UnaryOperator *unOp);
  mlir::Value generateBinaryOperator(clang::BinaryOperator *binOp);
};

} // namespace cmlirc

#endif // CMLIRC_ASTVISITOR_H
