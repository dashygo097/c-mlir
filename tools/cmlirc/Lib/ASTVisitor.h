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

  [[nodiscard]] ASTContext *Context() const { return Context_; }

private:
  ASTContext *Context_;
  unsigned IndentLevel_ = 0;

  void printIndent() {
    for (unsigned i = 0; i < IndentLevel_; ++i) {
      llvm::outs() << " ";
    }
  }
  void increaseIndent() { IndentLevel_ += 2; }
  void decreaseIndent() { IndentLevel_ -= 2; }
};

} // namespace cmlirc

#endif // CMLIRC_ASTVISITOR_H
