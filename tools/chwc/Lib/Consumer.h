#ifndef CHWC_ASTCONSUMER_H
#define CHWC_ASTCONSUMER_H

#include "../ArgumentList.h"
#include "./Converter.h"
#include "clang/AST/ASTConsumer.h"
#include "llvm/Support/WithColor.h"

namespace chwc {

class CHWConsumer : public clang::ASTConsumer {
public:
  explicit CHWConsumer(CHWContextManager &ctx) : visitor(ctx) {}
  ~CHWConsumer() override = default;

  void HandleTranslationUnit(clang::ASTContext &ctx) override {
    visitor.TraverseDecl(ctx.getTranslationUnitDecl());
  }

private:
  CHWConverter visitor;
};

} // namespace chwc

#endif // CHWC_ASTCONSUMER_H
