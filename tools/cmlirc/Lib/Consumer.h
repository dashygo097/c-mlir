#ifndef CMLIRC_ASTCONSUMER_H
#define CMLIRC_ASTCONSUMER_H

#include "../ArgumentList.h"
#include "./Converter.h"
#include "./Pragmas/PragmaHandler.h"
#include "clang/AST/ASTConsumer.h"
#include "clang/AST/RecursiveASTVisitor.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/WithColor.h"

namespace cmlirc {

class CMLIRFunctionCallCollector
    : public clang::RecursiveASTVisitor<CMLIRFunctionCallCollector> {
public:
  auto VisitCallExpr(clang::CallExpr *callExpr) -> bool {
    if (!callExpr) {
      return true;
    }

    clang::FunctionDecl *callee = callExpr->getDirectCallee();
    if (!callee) {
      return true;
    }

    clang::FunctionDecl *definition = callee->getDefinition();
    if (!definition) {
      return true;
    }

    callees.push_back(definition);
    return true;
  }

  auto getCallees() const -> llvm::ArrayRef<clang::FunctionDecl *> {
    return callees;
  }

private:
  llvm::SmallVector<clang::FunctionDecl *, 16> callees;
};

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

    clang::FunctionDecl *targetFunc =
        findFunctionDefinition(ctx.getTranslationUnitDecl(), targetFuncName);

    if (!targetFunc) {
      llvm::WithColor::error()
          << "cmlirc: function '" << targetFuncName << "' not found\n";
      return;
    }

    llvm::DenseSet<const clang::FunctionDecl *> visiting;
    llvm::DenseSet<const clang::FunctionDecl *> emitted;

    emitFunctionWithReachableCallees(targetFunc, visiting, emitted);
  }

private:
  auto getCanonicalFunctionDecl(clang::FunctionDecl *funcDecl)
      -> const clang::FunctionDecl * {
    if (!funcDecl) {
      return nullptr;
    }

    clang::FunctionDecl *definition = funcDecl->getDefinition();
    if (definition) {
      return definition->getCanonicalDecl();
    }

    return funcDecl->getCanonicalDecl();
  }

  auto findFunctionDefinition(clang::TranslationUnitDecl *tuDecl,
                              llvm::StringRef functionName)
      -> clang::FunctionDecl * {
    if (!tuDecl) {
      return nullptr;
    }

    for (clang::Decl *decl : tuDecl->decls()) {
      auto *funcDecl = llvm::dyn_cast<clang::FunctionDecl>(decl);
      if (!funcDecl) {
        continue;
      }

      if (funcDecl->getNameAsString() != functionName) {
        continue;
      }

      clang::FunctionDecl *definition = funcDecl->getDefinition();
      if (!definition) {
        llvm::WithColor::error() << "cmlirc: function '" << functionName
                                 << "' found but has no body\n";
        return nullptr;
      }

      return definition;
    }

    return nullptr;
  }

  void collectDirectDefinedCallees(
      clang::FunctionDecl *funcDecl,
      llvm::SmallVectorImpl<clang::FunctionDecl *> &callees) {
    if (!funcDecl || !funcDecl->hasBody()) {
      return;
    }

    CMLIRFunctionCallCollector collector;
    collector.TraverseStmt(funcDecl->getBody());

    for (clang::FunctionDecl *callee : collector.getCallees()) {
      if (!callee) {
        continue;
      }

      clang::FunctionDecl *definition = callee->getDefinition();
      if (!definition) {
        continue;
      }

      callees.push_back(definition);
    }
  }

  void emitFunctionWithReachableCallees(
      clang::FunctionDecl *funcDecl,
      llvm::DenseSet<const clang::FunctionDecl *> &visiting,
      llvm::DenseSet<const clang::FunctionDecl *> &emitted) {
    if (!funcDecl) {
      return;
    }

    clang::FunctionDecl *definition = funcDecl->getDefinition();
    if (!definition) {
      return;
    }

    const clang::FunctionDecl *canonical = getCanonicalFunctionDecl(definition);

    if (!canonical || emitted.contains(canonical)) {
      return;
    }

    if (visiting.contains(canonical)) {
      return;
    }

    visiting.insert(canonical);

    llvm::SmallVector<clang::FunctionDecl *, 16> callees;
    collectDirectDefinedCallees(definition, callees);

    for (clang::FunctionDecl *callee : callees) {
      emitFunctionWithReachableCallees(callee, visiting, emitted);
    }

    visiting.erase(canonical);

    visitor.TraverseFunctionDecl(definition);
    emitted.insert(canonical);
  }

  CMLIRConverter visitor;
};

} // namespace cmlirc

#endif // CMLIRC_ASTCONSUMER_H
