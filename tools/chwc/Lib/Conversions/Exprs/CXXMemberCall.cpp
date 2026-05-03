#include "../../Converter.h"
#include "../Utils/Annotation.h"
#include "llvm/Support/WithColor.h"

namespace chwc {

auto isMethodOfCurrentRecord(const clang::CXXRecordDecl *currentRecordDecl,
                             clang::CXXMethodDecl *methodDecl) -> bool {
  if (!currentRecordDecl || !methodDecl) {
    return false;
  }

  const clang::CXXRecordDecl *parent = methodDecl->getParent();
  if (!parent) {
    return false;
  }

  return parent->getCanonicalDecl() == currentRecordDecl->getCanonicalDecl();
}

auto CHWConverter::generateCXXMemberCallExpr(clang::CXXMemberCallExpr *callExpr)
    -> mlir::Value {
  clang::CXXMethodDecl *methodDecl = callExpr->getMethodDecl();
  if (!methodDecl) {
    llvm::WithColor::error()
        << "chwc: unsupported member call without resolved method\n";
    return nullptr;
  }

  if (!isMethodOfCurrentRecord(currentRecordDecl, methodDecl)) {
    llvm::WithColor::error()
        << "chwc: only calls to methods of the current hardware class are "
           "supported\n";
    return nullptr;
  }

  if (utils::isLifecycleMethod(methodDecl)) {
    llvm::WithColor::error()
        << "chwc: direct call to annotated lifecycle method is unsupported\n";
    return nullptr;
  }

  if (!methodDecl->hasBody()) {
    llvm::WithColor::error() << "chwc: member helper method has no body: "
                             << methodDecl->getNameAsString() << "\n";
    return nullptr;
  }

  if (methodDecl->getNumParams() != callExpr->getNumArgs()) {
    llvm::WithColor::error() << "chwc: member helper argument count mismatch: "
                             << methodDecl->getNameAsString() << "\n";
    return nullptr;
  }

  llvm::DenseMap<const clang::VarDecl *, mlir::Value> savedLocalValueTable =
      localValueTable;

  mlir::Value savedReturnValue = currentReturnValue;
  bool savedHasReturnValue = hasCurrentReturnValue;

  currentReturnValue = nullptr;
  hasCurrentReturnValue = false;
  ++helperInlineDepth;

  for (unsigned i = 0; i < callExpr->getNumArgs(); ++i) {
    mlir::Value argValue = generateExpr(callExpr->getArg(i));
    if (!argValue) {
      llvm::WithColor::error()
          << "chwc: failed to lower member helper argument " << i << "\n";

      --helperInlineDepth;
      currentReturnValue = savedReturnValue;
      hasCurrentReturnValue = savedHasReturnValue;
      localValueTable = std::move(savedLocalValueTable);
      return nullptr;
    }

    clang::ParmVarDecl *paramDecl = methodDecl->getParamDecl(i);
    localValueTable[paramDecl] = argValue;
  }

  TraverseStmt(methodDecl->getBody());

  mlir::Value returnValue = currentReturnValue;
  bool hasReturnValue = hasCurrentReturnValue;

  --helperInlineDepth;
  currentReturnValue = savedReturnValue;
  hasCurrentReturnValue = savedHasReturnValue;
  localValueTable = std::move(savedLocalValueTable);

  if (!hasReturnValue && !methodDecl->getReturnType()->isVoidType()) {
    llvm::WithColor::error() << "chwc: non-void member helper has no return: "
                             << methodDecl->getNameAsString() << "\n";
    return nullptr;
  }

  return returnValue;
}

} // namespace chwc
