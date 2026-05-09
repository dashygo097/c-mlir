#include "../../Converter.h"
#include "../Utils/Annotation.h"
#include "../Utils/Cast.h"
#include "../Utils/Comb.h"
#include "../Utils/State.h"
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

auto getFirstIntegralTemplateArg(clang::CallExpr *callExpr)
    -> std::optional<uint64_t> {
  if (!callExpr) {
    return std::nullopt;
  }

  clang::FunctionDecl *callee = callExpr->getDirectCallee();
  if (!callee) {
    return std::nullopt;
  }

  clang::FunctionTemplateSpecializationInfo *specInfo =
      callee->getTemplateSpecializationInfo();

  if (!specInfo || !specInfo->TemplateArguments) {
    return std::nullopt;
  }

  llvm::ArrayRef<clang::TemplateArgument> args =
      specInfo->TemplateArguments->asArray();

  if (args.empty()) {
    return std::nullopt;
  }

  const clang::TemplateArgument &arg = args.front();
  if (arg.getKind() != clang::TemplateArgument::Integral) {
    return std::nullopt;
  }

  return arg.getAsIntegral().getZExtValue();
}

auto CHWConverter::generateCallExpr(clang::CallExpr *callExpr) -> mlir::Value {
  if (!callExpr) {
    return nullptr;
  }

  std::string name = getCallName(callExpr);

  mlir::OpBuilder &builder = contextManager.Builder();
  mlir::Location loc = builder.getUnknownLoc();

  if (name == "Mux") {
    if (callExpr->getNumArgs() != 3) {
      llvm::WithColor::error() << "chwc: Mux expects 3 arguments\n";
      return nullptr;
    }

    mlir::Value cond = generateExpr(callExpr->getArg(0));
    mlir::Value trueValue = generateExpr(callExpr->getArg(1));
    mlir::Value falseValue = generateExpr(callExpr->getArg(2));

    if (!cond || !trueValue || !falseValue) {
      llvm::WithColor::error() << "chwc: failed to generate Mux operands\n";
      return nullptr;
    }

    cond = utils::toBool(builder, loc, cond);
    if (!cond) {
      return nullptr;
    }

    if (trueValue.getType() != falseValue.getType()) {
      falseValue =
          utils::promoteValue(builder, loc, falseValue, trueValue.getType());
      if (!falseValue) {
        return nullptr;
      }
    }

    return utils::mux(builder, loc, cond, trueValue, falseValue);
  }

  if (name == "RegNext") {
    if (callExpr->getNumArgs() != 1) {
      llvm::WithColor::error() << "chwc: RegNext expects 1 argument\n";
      return nullptr;
    }

    mlir::Value nextValue = generateExpr(callExpr->getArg(0));
    if (!nextValue) {
      llvm::WithColor::error() << "chwc: failed to generate RegNext input\n";
      return nullptr;
    }

    return utils::emitRegNext(moduleContext, builder, loc, nextValue);
  }

  if (name == "Delay") {
    if (callExpr->getNumArgs() != 1) {
      llvm::WithColor::error() << "chwc: Delay expects 1 argument\n";
      return nullptr;
    }

    std::optional<uint64_t> cycles = getFirstIntegralTemplateArg(callExpr);
    if (!cycles) {
      llvm::WithColor::error()
          << "chwc: Delay requires an integer template cycle count\n";
      return nullptr;
    }

    if (*cycles == 0) {
      llvm::WithColor::error() << "chwc: Delay cycle count must be >= 1\n";
      return nullptr;
    }

    if (*cycles > 1024) {
      llvm::WithColor::error() << "chwc: Delay cycle count is too large\n";
      return nullptr;
    }

    mlir::Value value = generateExpr(callExpr->getArg(0));
    if (!value) {
      llvm::WithColor::error() << "chwc: failed to generate Delay input\n";
      return nullptr;
    }

    return utils::emitDelay(moduleContext, builder, loc, value,
                            static_cast<unsigned>(*cycles));
  }

  clang::CXXMethodDecl *methodDecl = nullptr;

  if (!moduleContext.recordDecl) {
    llvm::WithColor::error()
        << "chwc: unsupported CallExpr outside module: " << name << "\n";
    return nullptr;
  }

  for (clang::CXXMethodDecl *method :
       const_cast<clang::CXXRecordDecl *>(moduleContext.recordDecl)
           ->methods()) {
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
