#include "./ASTVisitor.h"

namespace cmlirc {
using namespace clang;

CMLIRCASTVisitor::CMLIRCASTVisitor(ASTContext *Context) : Context_(Context) {}

bool CMLIRCASTVisitor::VisitFunctionDecl(FunctionDecl *FD) {
  llvm::outs() << "Function: " << FD->getNameAsString() << "\n";
  FD->dump();
  return true;
}

bool CMLIRCASTVisitor::VisitVarDecl(VarDecl *VD) {
  llvm::outs() << "Variable: " << VD->getNameAsString() << "\n";
  return true;
}

}; // namespace cmlirc
