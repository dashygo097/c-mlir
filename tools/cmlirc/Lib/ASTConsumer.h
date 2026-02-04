#include "./ASTVisitor.h"
#include "clang/AST/ASTConsumer.h"

namespace cmlirc {
using namespace clang;

class CMLIRCASTConsumer : public ASTConsumer {
public:
  explicit CMLIRCASTConsumer(ASTContext *Context) : Visitor(Context) {}

  void HandleTranslationUnit(ASTContext &Context) override {
    Context.getTranslationUnitDecl()->dump();
  }

private:
  CMLIRCASTVisitor Visitor;
};

} // namespace cmlirc
