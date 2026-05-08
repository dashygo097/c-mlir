#include "../../Converter.h"
#include "../Utils/Annotation.h"
#include "../Utils/Cast.h"
#include "../Utils/Type.h"
#include "llvm/Support/WithColor.h"

namespace chwc {

auto getCallName(clang::CallExpr *callExpr) -> std::string {
  if (!callExpr) {
    return "";
  }

  clang::Expr *callee = callExpr->getCallee();
  if (!callee) {
    return "";
  }

  callee = callee->IgnoreParenImpCasts();

  if (auto *declRef = llvm::dyn_cast<clang::DeclRefExpr>(callee)) {
    return declRef->getDecl()->getNameAsString();
  }

  if (auto *memberExpr = llvm::dyn_cast<clang::MemberExpr>(callee)) {
    return memberExpr->getMemberDecl()->getNameAsString();
  }

  if (auto *unresolved = llvm::dyn_cast<clang::UnresolvedLookupExpr>(callee)) {
    if (unresolved->getNumDecls() == 1) {
      return (*unresolved->decls_begin())->getNameAsString();
    }
  }

  return "";
}

auto CHWConverter::generateCallExpr(clang::CallExpr *callExpr) -> mlir::Value {
  std::string name = getCallName(callExpr);
  if (name.empty()) {
    llvm::WithColor::error() << "chwc: unsupported CallExpr callee\n";
    return nullptr;
  }

  clang::CXXMethodDecl *methodDecl = nullptr;

  for (clang::CXXMethodDecl *method : moduleContext.recordDecl->methods()) {
    if (method->getNameAsString() == name && utils::isFuncMethod(method)) {
      methodDecl = method;
      break;
    }
  }

  if (!methodDecl) {
    llvm::WithColor::error()
        << "chwc: unresolved HW_FUNC call: " << name << "\n";
    return nullptr;
  }

  if (!methodDecl->hasBody()) {
    llvm::WithColor::error()
        << "chwc: HW_FUNC method has no body: " << name << "\n";
    return nullptr;
  }

  if (methodDecl->getNumParams() != callExpr->getNumArgs()) {
    llvm::WithColor::error()
        << "chwc: HW_FUNC argument count mismatch: " << name << "\n";
    return nullptr;
  }

  functionStack.emplace_back();

  mlir::OpBuilder &builder = contextManager.Builder();
  mlir::Location loc = builder.getUnknownLoc();

  for (unsigned i = 0; i < callExpr->getNumArgs(); ++i) {
    mlir::Value argValue = generateExpr(callExpr->getArg(i));
    if (!argValue) {
      functionStack.pop_back();
      return nullptr;
    }

    clang::ParmVarDecl *paramDecl = methodDecl->getParamDecl(i);
    mlir::Type paramType = convertType(paramDecl->getType());
    if (!paramType) {
      functionStack.pop_back();
      return nullptr;
    }

    if (argValue.getType() != paramType) {
      argValue = utils::promoteValue(builder, loc, argValue, paramType);
      if (!argValue) {
        functionStack.pop_back();
        return nullptr;
      }
    }

    functionStack.back().locals[paramDecl] = argValue;
  }

  TraverseStmt(methodDecl->getBody());

  mlir::Value returnValue = functionStack.back().returnValue;
  bool hasReturnValue = functionStack.back().hasReturnValue;

  functionStack.pop_back();

  if (methodDecl->getReturnType()->isVoidType()) {
    return nullptr;
  }

  if (!hasReturnValue || !returnValue) {
    llvm::WithColor::error()
        << "chwc: non-void HW_FUNC has no return: " << name << "\n";
    return nullptr;
  }

  mlir::Type returnType = convertType(methodDecl->getReturnType());
  if (!returnType) {
    return nullptr;
  }

  if (returnValue.getType() != returnType) {
    returnValue = utils::promoteValue(builder, loc, returnValue, returnType);
  }

  return returnValue;
}

} // namespace chwc
