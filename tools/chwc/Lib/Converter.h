#ifndef CHWC_ASTVISITOR_H
#define CHWC_ASTVISITOR_H

#include "./Context/ContextManager.h"
#include "clang/AST/RecursiveASTVisitor.h"

namespace chwc {

class CHWConverter : public clang::RecursiveASTVisitor<CHWConverter> {
public:
  explicit CHWConverter(ContextManager &contextManager)
      : contextManager(contextManager) {}
  ~CHWConverter() = default;

  // decl traits

  // stmt traits

  // control flow

  // loop

  // loop optimizations

private:
  ContextManager &contextManager;

  // states

  // helpers

  // type traits

  // expr traits
};

} // namespace chwc

#endif // CHWC_ASTVISITOR_H
