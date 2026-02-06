#ifndef CMLIRC_ASTCONSUMER_H
#define CMLIRC_ASTCONSUMER_H

#include "./ASTVisitor.h"
#include "clang/AST/ASTConsumer.h"

namespace cmlirc {

class CMLIRCASTConsumer : public clang::ASTConsumer {
public:
  explicit CMLIRCASTConsumer(ContextManager &mlirContext)
      : visitor_(mlirContext) {}
  ~CMLIRCASTConsumer() = default;

  void HandleTranslationUnit(clang::ASTContext &Context) override {
    visitor_.TraverseDecl(Context.getTranslationUnitDecl());
  }

private:
  CMLIRCASTVisitor visitor_;
};

} // namespace cmlirc

#endif // CMLIRC_ASTCONSUMER_H
