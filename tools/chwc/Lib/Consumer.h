#ifndef CHWC_ASTCONSUMER_H
#define CHWC_ASTCONSUMER_H

#include "../ArgumentList.h"
#include "./Converter.h"
#include "clang/AST/ASTConsumer.h"
#include "clang/AST/DeclCXX.h"
#include "clang/AST/DeclTemplate.h"
#include "clang/Basic/SourceManager.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/WithColor.h"
#include <string>

namespace chwc {

class CHWConsumer : public clang::ASTConsumer {
public:
  explicit CHWConsumer(CHWContextManager &ctx) : visitor(ctx) {}
  ~CHWConsumer() override = default;

  void HandleTranslationUnit(clang::ASTContext &ctx) override {
    clang::TranslationUnitDecl *tuDecl = ctx.getTranslationUnitDecl();
    std::string targetModuleName = options::moduleName;

    bool foundTarget = targetModuleName.empty();
    scanDeclContext(ctx, tuDecl, targetModuleName, foundTarget);

    if (!targetModuleName.empty() && !foundTarget) {
      llvm::WithColor::error()
          << "chwc: module '" << targetModuleName << "' not found\n";
    }
  }

private:
  CHWConverter visitor;

  auto isFromMainFile(clang::ASTContext &ctx, clang::Decl *decl) -> bool {
    if (!decl) {
      return false;
    }

    clang::SourceLocation loc = decl->getLocation();
    if (loc.isInvalid()) {
      return false;
    }

    clang::SourceManager &sm = ctx.getSourceManager();
    loc = sm.getExpansionLoc(loc);

    return sm.isWrittenInMainFile(loc);
  }

  auto recordNameMatches(clang::CXXRecordDecl *recordDecl,
                         llvm::StringRef targetModuleName) -> bool {
    if (!recordDecl || targetModuleName.empty()) {
      return false;
    }

    if (recordDecl->getNameAsString() == targetModuleName) {
      return true;
    }

    if (recordDecl->getQualifiedNameAsString() == targetModuleName) {
      return true;
    }

    return false;
  }

  auto templateNameMatches(clang::ClassTemplateDecl *templateDecl,
                           llvm::StringRef targetModuleName) -> bool {
    if (!templateDecl || targetModuleName.empty()) {
      return false;
    }

    if (templateDecl->getNameAsString() == targetModuleName) {
      return true;
    }

    if (templateDecl->getQualifiedNameAsString() == targetModuleName) {
      return true;
    }

    clang::CXXRecordDecl *recordDecl = templateDecl->getTemplatedDecl();
    return recordNameMatches(recordDecl, targetModuleName);
  }

  void scanDeclContext(clang::ASTContext &ctx, clang::DeclContext *declContext,
                       llvm::StringRef targetModuleName, bool &foundTarget) {
    if (!declContext) {
      return;
    }

    for (clang::Decl *decl : declContext->decls()) {
      if (!decl) {
        continue;
      }

      if (auto *namespaceDecl = llvm::dyn_cast<clang::NamespaceDecl>(decl)) {
        if (isFromMainFile(ctx, namespaceDecl)) {
          scanDeclContext(ctx, namespaceDecl, targetModuleName, foundTarget);
        }
        continue;
      }

      if (auto *templateDecl = llvm::dyn_cast<clang::ClassTemplateDecl>(decl)) {
        if (!isFromMainFile(ctx, templateDecl)) {
          continue;
        }

        clang::CXXRecordDecl *recordDecl = templateDecl->getTemplatedDecl();
        if (!recordDecl || !recordDecl->isCompleteDefinition()) {
          continue;
        }

        if (templateNameMatches(templateDecl, targetModuleName)) {
          foundTarget = true;
        }

        visitor.TraverseCXXRecordDecl(recordDecl);
        continue;
      }

      auto *recordDecl = llvm::dyn_cast<clang::CXXRecordDecl>(decl);
      if (!recordDecl) {
        continue;
      }

      if (!isFromMainFile(ctx, recordDecl)) {
        continue;
      }

      if (!recordDecl->isCompleteDefinition()) {
        continue;
      }

      if (recordNameMatches(recordDecl, targetModuleName)) {
        foundTarget = true;
      }

      visitor.TraverseCXXRecordDecl(recordDecl);
    }
  }
};

} // namespace chwc

#endif // CHWC_ASTCONSUMER_H
