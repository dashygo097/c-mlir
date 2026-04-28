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
    std::string targetModuleName = options::moduleName;

    if (targetModuleName.empty()) {
      visitor.TraverseDecl(ctx.getTranslationUnitDecl());
      return;
    }

    clang::TranslationUnitDecl *tuDecl = ctx.getTranslationUnitDecl();
    bool found = false;

    for (auto *decl : tuDecl->decls()) {
      auto *recordDecl = mlir::dyn_cast<clang::CXXRecordDecl>(decl);
      if (!recordDecl) {
        continue;
      }

      if (recordDecl->getNameAsString() != targetModuleName) {
        continue;
      }

      found = true;

      if (!recordDecl->isCompleteDefinition()) {
        llvm::WithColor::error() << "chwc: module '" << targetModuleName
                                 << "' found but has no complete definition\n";
        return;
      }

      visitor.TraverseCXXRecordDecl(recordDecl);
      return;
    }

    if (!found) {
      llvm::WithColor::error()
          << "chwc: module '" << targetModuleName << "' not found\n";
    }
  }

private:
  CHWConverter visitor;
};

} // namespace chwc

#endif // CHWC_ASTCONSUMER_H
