#ifndef CMLIRC_ASTCONSUMER_H
#define CMLIRC_ASTCONSUMER_H

#include "../ArgumentList.h"
#include "./Converter.h"
#include "./Pragmas/PragmaHandler.h"
#include "clang/AST/ASTConsumer.h"
#include "llvm/Support/WithColor.h"

namespace cmlirc {

class CMLIRConsumer : public clang::ASTConsumer {
public:
  explicit CMLIRConsumer(CMLIRContextManager &ctx, LoopHintMap &loopHintMap)
      : visitor(ctx, loopHintMap) {}
  ~CMLIRConsumer() override = default;

  void HandleTranslationUnit(clang::ASTContext &ctx) override {
    std::string targetFuncName = options::functionName;

    if (targetFuncName.empty()) {
      visitor.TraverseDecl(ctx.getTranslationUnitDecl());
      return;
    }

    clang::TranslationUnitDecl *tuDecl = ctx.getTranslationUnitDecl();
    bool found = false;

    for (auto *decl : tuDecl->decls()) {
      if (auto *funcDecl = mlir::dyn_cast<clang::FunctionDecl>(decl)) {
        if (funcDecl->getNameAsString() == targetFuncName) {
          if (!funcDecl->hasBody()) {
            llvm::WithColor::error() << "cmlirc: function '" << targetFuncName
                                     << "' found but has no body\n";
            return;
          }

          visitor.TraverseFunctionDecl(funcDecl);
          found = true;
          break;
        }
      }
    }

    if (!found) {
      llvm::WithColor::error()
          << "cmlirc: function '" << targetFuncName << "' not found\n";
    }
  }

private:
  CMLIRConverter visitor;
};

} // namespace cmlirc

#endif // CMLIRC_ASTCONSUMER_H
