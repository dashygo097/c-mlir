#include "../../Converter.h"
#include "../Utils/Annotation.h"
#include "../Utils/Cast.h"
#include "../Utils/Type.h"
#include "clang/AST/DeclCXX.h"
#include "llvm/Support/WithColor.h"

namespace chwc {

auto isCurrentModuleMethod(const clang::CXXRecordDecl *recordDecl,
                           clang::CXXMethodDecl *methodDecl) -> bool {
  if (!recordDecl || !methodDecl || !methodDecl->getParent()) {
    return false;
  }

  return methodDecl->getParent()->getCanonicalDecl() ==
         recordDecl->getCanonicalDecl();
}

auto isSignalReadMethod(clang::CXXMethodDecl *methodDecl) -> bool {
  if (!methodDecl) {
    return false;
  }

  std::string name = methodDecl->getNameAsString();

  if (name == "read" || name == "value" || name == "raw") {
    return true;
  }

  return mlir::isa<clang::CXXConversionDecl>(methodDecl);
}

auto isSignalBoolMethod(clang::CXXMethodDecl *methodDecl) -> bool {
  auto *conversion =
      mlir::dyn_cast_or_null<clang::CXXConversionDecl>(methodDecl);

  return conversion && conversion->getConversionType()->isBooleanType();
}

auto CHWConverter::generateCXXMemberCallExpr(clang::CXXMemberCallExpr *callExpr)
    -> mlir::Value {
  clang::CXXMethodDecl *methodDecl = callExpr->getMethodDecl();
  if (!methodDecl) {
    llvm::WithColor::error()
        << "chwc: unsupported member call without resolved method\n";
    return nullptr;
  }

  clang::Expr *objectExpr = callExpr->getImplicitObjectArgument();
  clang::QualType objectType =
      objectExpr ? objectExpr->getType() : clang::QualType{};

  if (!objectType.isNull() && utils::isSignalType(objectType)) {
    if (!isSignalReadMethod(methodDecl)) {
      llvm::WithColor::error() << "chwc: unsupported Signal member call: "
                               << methodDecl->getNameAsString() << "\n";
      return nullptr;
    }

    mlir::Value value = generateExpr(objectExpr);
    if (!value) {
      return nullptr;
    }

    mlir::OpBuilder &builder = contextManager.Builder();
    mlir::Location loc = builder.getUnknownLoc();

    if (isSignalBoolMethod(methodDecl)) {
      return utils::toBool(builder, loc, value);
    }

    mlir::Type targetType = convertType(callExpr->getType());
    if (targetType) {
      value = utils::promoteValue(builder, loc, value, targetType);
    }

    return value;
  }

  if (!isCurrentModuleMethod(moduleContext.recordDecl, methodDecl)) {
    llvm::WithColor::error()
        << "chwc: only calls to methods of the current module class are "
           "supported\n";
    return nullptr;
  }

  if (utils::isLifecycleMethod(methodDecl)) {
    llvm::WithColor::error()
        << "chwc: direct call to lifecycle method is unsupported\n";
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
        << "chwc: HW_FUNC method has no body: " << methodDecl->getNameAsString()
        << "\n";
    return nullptr;
  }

  if (methodDecl->getNumParams() != callExpr->getNumArgs()) {
    llvm::WithColor::error() << "chwc: HW_FUNC argument count mismatch: "
                             << methodDecl->getNameAsString() << "\n";
    return nullptr;
  }

  if (!methodDecl->getReturnType()->isVoidType() &&
      !utils::isValueType(methodDecl->getReturnType())) {
    llvm::WithColor::error()
        << "chwc: HW_FUNC return type must be UInt<W>, SInt<W>, or void\n";
    return nullptr;
  }

  for (clang::ParmVarDecl *paramDecl : methodDecl->parameters()) {
    if (!utils::isValueType(paramDecl->getType())) {
      llvm::WithColor::error()
          << "chwc: HW_FUNC parameter type must be UInt<W> or SInt<W>\n";
      return nullptr;
    }
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

    argValue = utils::promoteValue(builder, loc, argValue, paramType);
    if (!argValue) {
      functionStack.pop_back();
      return nullptr;
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
    llvm::WithColor::error() << "chwc: non-void HW_FUNC has no return: "
                             << methodDecl->getNameAsString() << "\n";
    return nullptr;
  }

  mlir::Type returnType = convertType(methodDecl->getReturnType());
  if (!returnType) {
    return nullptr;
  }

  return utils::promoteValue(builder, loc, returnValue, returnType);
}

} // namespace chwc
