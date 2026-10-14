#include "clang-support/Discovery.h"
#include "clang/AST/RecursiveASTVisitor.h"
#include "clang/Basic/SourceManager.h"
#include "llvm/ADT/DenseSet.h"

namespace clang_support {

namespace {

class Functions : public clang::RecursiveASTVisitor<Functions> {
public:
  explicit Functions(clang::ASTContext &ast) : ast(ast) {}

  llvm::SmallVector<clang::FunctionDecl *> definitions;

  auto VisitFunctionDecl(clang::FunctionDecl *decl) -> bool {
    if (decl->isThisDeclarationADefinition() && !decl->isImplicit() &&
        !ast.getSourceManager().isInSystemHeader(decl->getLocation()) &&
        seen.insert(decl->getCanonicalDecl()).second) {
      definitions.push_back(decl);
    }
    return true;
  }

private:
  clang::ASTContext &ast;
  llvm::DenseSet<const clang::FunctionDecl *> seen;
};

class Callees : public clang::RecursiveASTVisitor<Callees> {
public:
  llvm::SmallVector<clang::FunctionDecl *> definitions;

  auto VisitCallExpr(clang::CallExpr *expr) -> bool {
    if (auto *callee = expr->getDirectCallee()) {
      if (auto *definition = callee->getDefinition()) {
        if (seen.insert(definition->getCanonicalDecl()).second) {
          definitions.push_back(definition);
        }
      }
    }
    return true;
  }

private:
  llvm::DenseSet<const clang::FunctionDecl *> seen;
};

} // namespace

auto definedFunctions(clang::ASTContext &ast)
    -> llvm::SmallVector<clang::FunctionDecl *> {
  Functions visitor(ast);
  visitor.TraverseDecl(ast.getTranslationUnitDecl());
  return std::move(visitor.definitions);
}

auto directDefinedCallees(clang::FunctionDecl &function)
    -> llvm::SmallVector<clang::FunctionDecl *> {
  Callees visitor;
  visitor.TraverseStmt(function.getBody());
  return std::move(visitor.definitions);
}

} // namespace clang_support
