#include "../../Converter.h"
#include "../Utils/Annotation.h"
#include "../Utils/Cast.h"
#include "../Utils/Expr.h"
#include "../Utils/Type.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/WithColor.h"

namespace chwc {

auto callExprMethodOfCurrentRecord(
    const clang::CXXRecordDecl *currentRecordDecl,
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

auto callExprValidHelperValueType(clang::QualType type) -> bool {
  utils::SignalTypeInfo typeInfo = utils::getSignalTypeInfo(type);
  return typeInfo.isValue && !typeInfo.isSignal;
}

auto callExprGetMethodDecl(clang::CallExpr *callExpr)
    -> clang::CXXMethodDecl * {
  if (!callExpr) {
    return nullptr;
  }

  if (clang::FunctionDecl *directCallee = callExpr->getDirectCallee()) {
    if (auto *methodDecl = llvm::dyn_cast<clang::CXXMethodDecl>(directCallee)) {
      return methodDecl;
    }
  }

  clang::Expr *calleeExpr = callExpr->getCallee();
  calleeExpr = utils::ignoreExprWrappers(calleeExpr);

  if (auto *memberExpr =
          llvm::dyn_cast_or_null<clang::MemberExpr>(calleeExpr)) {
    return llvm::dyn_cast<clang::CXXMethodDecl>(memberExpr->getMemberDecl());
  }

  if (auto *declRef = llvm::dyn_cast_or_null<clang::DeclRefExpr>(calleeExpr)) {
    return llvm::dyn_cast<clang::CXXMethodDecl>(declRef->getDecl());
  }

  return nullptr;
}

auto CHWConverter::generateCallExpr(clang::CallExpr *callExpr) -> mlir::Value {
  if (!callExpr) {
    return nullptr;
  }

  clang::CXXMethodDecl *methodDecl = callExprGetMethodDecl(callExpr);
  if (!methodDecl) {
    llvm::WithColor::error()
        << "chwc: unsupported CallExpr without current-class method callee\n";
    return nullptr;
  }

  if (!callExprMethodOfCurrentRecord(currentRecordDecl, methodDecl)) {
    llvm::WithColor::error()
        << "chwc: only calls to methods of the current hardware class are "
           "supported: "
        << methodDecl->getQualifiedNameAsString() << "\n";
    return nullptr;
  }

  if (utils::isLifecycleMethod(methodDecl)) {
    llvm::WithColor::error()
        << "chwc: direct call to annotated lifecycle method is unsupported\n";
    return nullptr;
  }

  if (!utils::isFuncMethod(methodDecl)) {
    llvm::WithColor::error()
        << "chwc: helper method must be annotated with HW_FUNC: "
        << methodDecl->getNameAsString() << "\n";
    return nullptr;
  }

  if (!methodDecl->hasBody()) {
    llvm::WithColor::error()
        << "chwc: helper method has no body: " << methodDecl->getNameAsString()
        << "\n";
    return nullptr;
  }

  if (!methodDecl->getReturnType()->isVoidType() &&
      !callExprValidHelperValueType(methodDecl->getReturnType())) {
    llvm::WithColor::error()
        << "chwc: HW_FUNC return type must be UInt<W>, SInt<W> or void: "
        << methodDecl->getNameAsString() << "\n";
    return nullptr;
  }

  if (methodDecl->getNumParams() != callExpr->getNumArgs()) {
    llvm::WithColor::error() << "chwc: helper argument count mismatch: "
                             << methodDecl->getNameAsString() << "\n";
    return nullptr;
  }

  for (clang::ParmVarDecl *paramDecl : methodDecl->parameters()) {
    if (!callExprValidHelperValueType(paramDecl->getType())) {
      llvm::WithColor::error()
          << "chwc: HW_FUNC parameter type must be UInt<W> or SInt<W>: "
          << methodDecl->getNameAsString() << "\n";
      return nullptr;
    }
  }

  llvm::DenseMap<const clang::VarDecl *, mlir::Value> savedLocalValueTable =
      localValueTable;
  llvm::DenseMap<const clang::VarDecl *, int64_t> savedLocalConstIntTable =
      localConstIntTable;

  mlir::Value savedReturnValue = currentReturnValue;
  bool savedHasReturnValue = hasCurrentReturnValue;

  currentReturnValue = nullptr;
  hasCurrentReturnValue = false;
  ++helperInlineDepth;

  mlir::OpBuilder &builder = contextManager.Builder();
  mlir::Location loc = builder.getUnknownLoc();

  for (unsigned i = 0; i < callExpr->getNumArgs(); ++i) {
    mlir::Value argValue = generateExpr(callExpr->getArg(i));
    if (!argValue) {
      llvm::WithColor::error()
          << "chwc: failed to lower helper argument " << i << "\n";

      --helperInlineDepth;
      currentReturnValue = savedReturnValue;
      hasCurrentReturnValue = savedHasReturnValue;
      localValueTable = std::move(savedLocalValueTable);
      localConstIntTable = std::move(savedLocalConstIntTable);
      return nullptr;
    }

    clang::ParmVarDecl *paramDecl = methodDecl->getParamDecl(i);
    mlir::Type paramType = convertType(paramDecl->getType());
    if (!paramType) {
      --helperInlineDepth;
      currentReturnValue = savedReturnValue;
      hasCurrentReturnValue = savedHasReturnValue;
      localValueTable = std::move(savedLocalValueTable);
      localConstIntTable = std::move(savedLocalConstIntTable);
      return nullptr;
    }

    argValue = utils::promoteValue(builder, loc, argValue, paramType);
    if (!argValue) {
      --helperInlineDepth;
      currentReturnValue = savedReturnValue;
      hasCurrentReturnValue = savedHasReturnValue;
      localValueTable = std::move(savedLocalValueTable);
      localConstIntTable = std::move(savedLocalConstIntTable);
      return nullptr;
    }

    localValueTable[paramDecl] = argValue;
  }

  TraverseStmt(methodDecl->getBody());

  mlir::Value returnValue = currentReturnValue;
  bool hasReturnValue = hasCurrentReturnValue;

  --helperInlineDepth;
  currentReturnValue = savedReturnValue;
  hasCurrentReturnValue = savedHasReturnValue;
  localValueTable = std::move(savedLocalValueTable);
  localConstIntTable = std::move(savedLocalConstIntTable);

  if (methodDecl->getReturnType()->isVoidType()) {
    return nullptr;
  }

  if (!hasReturnValue) {
    llvm::WithColor::error() << "chwc: non-void helper has no return: "
                             << methodDecl->getNameAsString() << "\n";
    return nullptr;
  }

  mlir::Type returnType = convertType(methodDecl->getReturnType());
  if (!returnType) {
    return nullptr;
  }

  returnValue = utils::promoteValue(builder, loc, returnValue, returnType);
  if (!returnValue) {
    return nullptr;
  }

  return returnValue;
}

} // namespace chwc
