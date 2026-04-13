#ifndef CHWC_ASTCONSUMER_H
#define CHWC_ASTCONSUMER_H

#include "../ArgumentList.h"
#include "./Converter.h"
#include "clang/AST/ASTConsumer.h"

namespace chwc {

class CHWConsumer : public clang::ASTConsumer {
public:
  explicit CHWConsumer(ContextManager &ctx) : visitor(ctx) {}
  ~CHWConsumer() override = default;

  void HandleTranslationUnit(clang::ASTContext &ctx) override {}

private:
  CHWConverter visitor;
};

} // namespace chwc

#endif // CHWC_ASTCONSUMER_H
