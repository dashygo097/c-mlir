#include "clang/AST/RecursiveASTVisitor.h"

namespace cmlirc {
using namespace clang;

class CMLIRCASTVisitor : public clang::RecursiveASTVisitor<CMLIRCASTVisitor> {
public:
  explicit CMLIRCASTVisitor(ASTContext *Context);

  bool VisitFunctionDecl(FunctionDecl *FD);
  bool VisitVarDecl(VarDecl *VD);

public:
  ASTContext *Context_;
};

} // namespace cmlirc
