#ifndef CMLIRC_ASTCONSUMER_H
#define CMLIRC_ASTCONSUMER_H

#include "./ASTVisitor.h"
#include "clang/AST/ASTConsumer.h"

namespace cmlirc {
using namespace clang;

class CMLIRCASTConsumer : public ASTConsumer {
public:
  explicit CMLIRCASTConsumer(ASTContext *Context) : Visitor(Context) {}

  void HandleTranslationUnit(ASTContext &Context) override {
    Context.getTranslationUnitDecl()->dump();
    // Visitor_.TraverseDecl(Context.getTranslationUnitDecl());
  }

private:
  CMLIRCASTVisitor Visitor;
};

} // namespace cmlirc

#endif // CMLIRC_ASTCONSUMER_H
