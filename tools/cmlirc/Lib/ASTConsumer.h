#ifndef CMLIRC_ASTCONSUMER_H
#define CMLIRC_ASTCONSUMER_H

#include "../ArgumentList.h"
#include "./ASTVisitor.h"
#include "clang/AST/ASTConsumer.h"

namespace cmlirc {

class CMLIRCASTConsumer : public clang::ASTConsumer {
public:
  explicit CMLIRCASTConsumer(ContextManager &mlirContext)
      : visitor_(mlirContext) {}
  ~CMLIRCASTConsumer() = default;

  void HandleTranslationUnit(clang::ASTContext &Context) override {
    std::string targetFuncName = options::FunctionName;

    if (targetFuncName.empty()) {
      visitor_.TraverseDecl(Context.getTranslationUnitDecl());
      return;
    }

    clang::TranslationUnitDecl *TU = Context.getTranslationUnitDecl();
    bool found = false;

    for (auto *Decl : TU->decls()) {
      if (auto *FD = llvm::dyn_cast<clang::FunctionDecl>(Decl)) {
        if (FD->getNameAsString() == targetFuncName) {
          if (!FD->hasBody()) {
            llvm::errs() << "Error: Function '" << targetFuncName
                         << "' found but has no body\n";
            return;
          }

          visitor_.TraverseFunctionDecl(FD);
          found = true;
          break;
        }
      }
    }

    if (!found) {
      llvm::errs() << "Error: Function '" << targetFuncName << "' not found\n";
    }
  }

private:
  CMLIRCASTVisitor visitor_;
};

} // namespace cmlirc

#endif // CMLIRC_ASTCONSUMER_H
