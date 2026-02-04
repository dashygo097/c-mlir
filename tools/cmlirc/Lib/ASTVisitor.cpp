#include "./ASTVisitor.h"

namespace cmlirc {
using namespace clang;

CMLIRCASTVisitor::CMLIRCASTVisitor(ASTContext *Context) : Context(Context) {}

bool CMLIRCASTVisitor::VisitFunctionDecl(FunctionDecl *FD) { return true; }

bool CMLIRCASTVisitor::VisitVarDecl(VarDecl *VD) { return true; }

bool CMLIRCASTVisitor::VisitStmt(Stmt *S) { return true; }

bool CMLIRCASTVisitor::VisitBinaryOperator(BinaryOperator *BO) { return true; }

bool CMLIRCASTVisitor::VisitCallExpr(CallExpr *CE) { return true; }

} // namespace cmlirc
