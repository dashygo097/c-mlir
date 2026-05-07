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

namespace chwc {

class CHWConsumer : public clang::ASTConsumer {
public:
  explicit CHWConsumer(CHWContextManager &ctx) : visitor(ctx) {}
  ~CHWConsumer() override = default;

  void HandleTranslationUnit(clang::ASTContext &ctx) override {
    clang::TranslationUnitDecl *tuDecl = ctx.getTranslationUnitDecl();
    std::string targetModuleName = options::moduleName;

    bool found = false;
    scanDeclContext(ctx, tuDecl, targetModuleName, found);

    if (!targetModuleName.empty() && !found) {
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
    if (!recordDecl) {
      return false;
    }

    if (targetModuleName.empty()) {
      return true;
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
    if (!templateDecl) {
      return false;
    }

    if (targetModuleName.empty()) {
      return true;
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
                       llvm::StringRef targetModuleName, bool &found) {
    if (!declContext) {
      return;
    }

    for (clang::Decl *decl : declContext->decls()) {
      if (!decl) {
        continue;
      }

      if (auto *namespaceDecl = llvm::dyn_cast<clang::NamespaceDecl>(decl)) {
        if (isFromMainFile(ctx, namespaceDecl)) {
          scanDeclContext(ctx, namespaceDecl, targetModuleName, found);
        }
        continue;
      }

      if (auto *templateDecl = llvm::dyn_cast<clang::ClassTemplateDecl>(decl)) {
        if (!isFromMainFile(ctx, templateDecl)) {
          continue;
        }

        if (!templateNameMatches(templateDecl, targetModuleName)) {
          continue;
        }

        clang::CXXRecordDecl *recordDecl = templateDecl->getTemplatedDecl();
        if (!recordDecl || !recordDecl->isCompleteDefinition()) {
          continue;
        }

        visitor.TraverseCXXRecordDecl(recordDecl);

        if (!targetModuleName.empty()) {
          found = true;
          return;
        }

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

      if (!recordNameMatches(recordDecl, targetModuleName)) {
        continue;
      }

      visitor.TraverseCXXRecordDecl(recordDecl);

      if (!targetModuleName.empty()) {
        found = true;
        return;
      }
    }
  }
};

} // namespace chwc

#endif // CHWC_ASTCONSUMER_H
