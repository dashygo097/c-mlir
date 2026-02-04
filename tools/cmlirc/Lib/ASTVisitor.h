#ifndef CMLIRC_ASTVISITOR_H
#define CMLIRC_ASTVISITOR_H

#include "clang/AST/RecursiveASTVisitor.h"

namespace cmlirc {
using namespace clang;

class CMLIRCASTVisitor : public RecursiveASTVisitor<CMLIRCASTVisitor> {
public:
  explicit CMLIRCASTVisitor(ASTContext *Context);

  bool VisitFunctionDecl(FunctionDecl *FD);
  bool VisitVarDecl(VarDecl *VD);
  bool VisitStmt(Stmt *S);
  bool VisitBinaryOperator(BinaryOperator *BO);
  bool VisitCallExpr(CallExpr *CE);

  ASTContext *Context;
};

} // namespace cmlirc

#endif // CMLIRC_ASTVISITOR_H
