#include "../../Converter.h"
#include "../Utils/Annotation.h"
#include "../Utils/Cast.h"
#include "../Utils/Type.h"
#include "clang/AST/DeclCXX.h"
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

auto getImplicitObject(clang::CXXMemberCallExpr *callExpr) -> clang::Expr * {
  if (!callExpr) {
    return nullptr;
  }

  return callExpr->getImplicitObjectArgument();
}

auto isSignalReadMethod(clang::CXXMethodDecl *methodDecl) -> bool {
  if (!methodDecl) {
    return false;
  }

  std::string name = methodDecl->getNameAsString();

  if (name == "read" || name == "value" || name == "raw") {
    return true;
  }

  if (mlir::isa<clang::CXXConversionDecl>(methodDecl)) {
    return true;
  }

  return false;
}

auto isSignalBoolMethod(clang::CXXMethodDecl *methodDecl) -> bool {
  if (!methodDecl) {
    return false;
  }

  auto *conversionDecl = mlir::dyn_cast<clang::CXXConversionDecl>(methodDecl);
  if (!conversionDecl) {
    return false;
  }

  return conversionDecl->getConversionType()->isBooleanType();
}

auto isValidHelperValueType(clang::QualType type) -> bool {
  utils::SignalTypeInfo typeInfo = utils::getSignalTypeInfo(type);
  return typeInfo.isValue && !typeInfo.isSignal;
}

auto CHWConverter::generateCXXMemberCallExpr(clang::CXXMemberCallExpr *callExpr)
    -> mlir::Value {
  clang::CXXMethodDecl *methodDecl = callExpr->getMethodDecl();
  if (!methodDecl) {
    llvm::WithColor::error()
        << "chwc: unsupported member call without resolved method\n";
    return nullptr;
  }

  clang::Expr *objectExpr = getImplicitObject(callExpr);
  clang::QualType objectType =
      objectExpr ? objectExpr->getType() : clang::QualType{};

  if (!objectType.isNull() && utils::isSignalType(objectType)) {
    if (isSignalReadMethod(methodDecl)) {
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
        if (!value) {
          return nullptr;
        }
      }

      return value;
    }

    llvm::WithColor::error() << "chwc: unsupported Signal member call: "
                             << methodDecl->getNameAsString() << "\n";
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

  if (!utils::isFuncMethod(methodDecl)) {
    llvm::WithColor::error()
        << "chwc: helper method must be annotated with HW_FUNC: "
        << methodDecl->getNameAsString() << "\n";
    return nullptr;
  }

  if (!methodDecl->hasBody()) {
    llvm::WithColor::error() << "chwc: member helper method has no body: "
                             << methodDecl->getNameAsString() << "\n";
    return nullptr;
  }

  if (!methodDecl->getReturnType()->isVoidType() &&
      !isValidHelperValueType(methodDecl->getReturnType())) {
    llvm::WithColor::error()
        << "chwc: HW_FUNC return type must be UInt<W> or void: "
        << methodDecl->getNameAsString() << "\n";
    return nullptr;
  }

  if (methodDecl->getNumParams() != callExpr->getNumArgs()) {
    llvm::WithColor::error() << "chwc: member helper argument count mismatch: "
                             << methodDecl->getNameAsString() << "\n";
    return nullptr;
  }

  for (clang::ParmVarDecl *paramDecl : methodDecl->parameters()) {
    if (!isValidHelperValueType(paramDecl->getType())) {
      llvm::WithColor::error()
          << "chwc: HW_FUNC parameter type must be UInt<W>: "
          << methodDecl->getNameAsString() << "\n";
      return nullptr;
    }
  }

  llvm::DenseMap<const clang::VarDecl *, mlir::Value> savedLocalValueTable =
      localValueTable;

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
          << "chwc: failed to lower member helper argument " << i << "\n";

      --helperInlineDepth;
      currentReturnValue = savedReturnValue;
      hasCurrentReturnValue = savedHasReturnValue;
      localValueTable = std::move(savedLocalValueTable);
      return nullptr;
    }

    clang::ParmVarDecl *paramDecl = methodDecl->getParamDecl(i);
    mlir::Type paramType = convertType(paramDecl->getType());
    if (!paramType) {
      --helperInlineDepth;
      currentReturnValue = savedReturnValue;
      hasCurrentReturnValue = savedHasReturnValue;
      localValueTable = std::move(savedLocalValueTable);
      return nullptr;
    }

    argValue = utils::promoteValue(builder, loc, argValue, paramType);
    if (!argValue) {
      --helperInlineDepth;
      currentReturnValue = savedReturnValue;
      hasCurrentReturnValue = savedHasReturnValue;
      localValueTable = std::move(savedLocalValueTable);
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

  if (!hasReturnValue && !methodDecl->getReturnType()->isVoidType()) {
    llvm::WithColor::error() << "chwc: non-void member helper has no return: "
                             << methodDecl->getNameAsString() << "\n";
    return nullptr;
  }

  if (methodDecl->getReturnType()->isVoidType()) {
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
