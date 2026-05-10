#ifndef CHWC_UTILS_TEMPLATE_H
#define CHWC_UTILS_TEMPLATE_H

#include "../../Converter.h"
#include "clang/AST/DeclTemplate.h"
#include "clang/AST/Expr.h"
#include "clang/AST/Type.h"
#include "llvm/Support/WithColor.h"
#include <optional>
#include <string>

namespace chwc::utils {

inline auto getTemplateName(clang::TemplateName name) -> std::string {
  if (auto *templateDecl = name.getAsTemplateDecl()) {
    return templateDecl->getNameAsString();
  }

  return "";
}

inline auto getRecordTemplateSpec(clang::QualType type)
    -> const clang::ClassTemplateSpecializationDecl * {
  type = type.getCanonicalType().getUnqualifiedType();

  const clang::Type *typePtr = type.getTypePtrOrNull();
  if (!typePtr) {
    return nullptr;
  }

  auto *recordType = typePtr->getAs<clang::RecordType>();
  if (!recordType) {
    return nullptr;
  }

  return llvm::dyn_cast<clang::ClassTemplateSpecializationDecl>(
      recordType->getDecl());
}

inline auto getTemplateSpecializationType(clang::QualType type)
    -> const clang::TemplateSpecializationType * {
  type = type.getCanonicalType().getUnqualifiedType();

  const clang::Type *typePtr = type.getTypePtrOrNull();
  if (!typePtr) {
    return nullptr;
  }

  return llvm::dyn_cast<clang::TemplateSpecializationType>(typePtr);
}

inline auto getTemplateSpecializationName(clang::QualType type) -> std::string {
  if (auto *spec = getRecordTemplateSpec(type)) {
    if (spec->getSpecializedTemplate()) {
      return spec->getSpecializedTemplate()->getNameAsString();
    }
  }

  if (auto *specType = getTemplateSpecializationType(type)) {
    return getTemplateName(specType->getTemplateName());
  }

  return "";
}

inline auto getTemplateArgCount(clang::QualType type) -> unsigned {
  if (auto *spec = getRecordTemplateSpec(type)) {
    return spec->getTemplateArgs().size();
  }

  if (auto *specType = getTemplateSpecializationType(type)) {
    return static_cast<unsigned>(specType->template_arguments().size());
  }

  return 0;
}

inline auto getTemplateArg(clang::QualType type, unsigned index)
    -> std::optional<clang::TemplateArgument> {
  if (auto *spec = getRecordTemplateSpec(type)) {
    const clang::TemplateArgumentList &args = spec->getTemplateArgs();
    if (index >= args.size()) {
      return std::nullopt;
    }

    return args[index];
  }

  if (auto *specType = getTemplateSpecializationType(type)) {
    llvm::ArrayRef<clang::TemplateArgument> args =
        specType->template_arguments();

    if (index >= args.size()) {
      return std::nullopt;
    }

    return args[index];
  }

  return std::nullopt;
}

inline auto getTemplateWidthNameFromExpr(clang::Expr *expr)
    -> std::optional<std::string> {
  if (!expr) {
    return std::nullopt;
  }

  expr = expr->IgnoreParenImpCasts();

  if (auto *declRef = llvm::dyn_cast<clang::DeclRefExpr>(expr)) {
    if (auto *parm = llvm::dyn_cast<clang::NonTypeTemplateParmDecl>(
            declRef->getDecl())) {
      return parm->getNameAsString();
    }
  }

  if (auto *subst = llvm::dyn_cast<clang::SubstNonTypeTemplateParmExpr>(expr)) {
    if (auto *parm = subst->getParameter()) {
      return parm->getNameAsString();
    }
  }

  return std::nullopt;
}

inline auto getFirstIntegralTemplateArg(clang::CallExpr *callExpr)
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

inline auto getTemplateIntegerDefaultAttr(mlir::OpBuilder &builder,
                                          clang::NonTypeTemplateParmDecl *param)
    -> mlir::Attribute {
  if (!param || !param->hasDefaultArgument()) {
    return {};
  }

  mlir::Type paramType = builder.getIntegerType(32);

  clang::TemplateArgumentLoc defaultArgLoc = param->getDefaultArgument();
  const clang::TemplateArgument &defaultArg = defaultArgLoc.getArgument();

  if (defaultArg.getKind() == clang::TemplateArgument::Integral) {
    return mlir::IntegerAttr::get(paramType,
                                  defaultArg.getAsIntegral().getSExtValue());
  }

  clang::Expr *defaultExpr = defaultArgLoc.getSourceExpression();
  if (!defaultExpr) {
    llvm::WithColor::error() << "chwc: unsupported template parameter default: "
                             << param->getNameAsString() << "\n";
    return {};
  }

  defaultExpr = defaultExpr->IgnoreParenImpCasts();

  auto *intLit = llvm::dyn_cast<clang::IntegerLiteral>(defaultExpr);
  if (!intLit) {
    llvm::WithColor::error()
        << "chwc: template parameter default must be integer literal: "
        << param->getNameAsString() << "\n";
    return {};
  }

  return mlir::IntegerAttr::get(paramType, intLit->getValue().getSExtValue());
}

inline void collectTemplateParameters(HWModuleContext &moduleContext,
                                      mlir::OpBuilder &builder,
                                      clang::CXXRecordDecl *recordDecl) {
  clang::ClassTemplateDecl *classTemplate =
      recordDecl->getDescribedClassTemplate();

  if (!classTemplate) {
    return;
  }

  clang::TemplateParameterList *params = classTemplate->getTemplateParameters();

  for (clang::NamedDecl *paramDecl : *params) {
    auto *nttp = llvm::dyn_cast<clang::NonTypeTemplateParmDecl>(paramDecl);
    if (!nttp) {
      llvm::WithColor::error()
          << "chwc: only non-type integer template parameters are supported\n";
      continue;
    }

    if (!nttp->getType()->isIntegerType()) {
      llvm::WithColor::error()
          << "chwc: template module parameter must be integer: "
          << nttp->getNameAsString() << "\n";
      continue;
    }

    mlir::Type paramType = builder.getIntegerType(32);
    mlir::StringAttr nameAttr = builder.getStringAttr(nttp->getNameAsString());
    mlir::Attribute defaultAttr = getTemplateIntegerDefaultAttr(builder, nttp);

    mlir::Attribute paramDeclAttr = circt::hw::ParamDeclAttr::get(
        builder.getContext(), nameAttr, paramType, defaultAttr);

    mlir::TypedAttr paramRefAttr = circt::hw::ParamDeclRefAttr::get(
        builder.getContext(), nameAttr, paramType);

    moduleContext.parameters.push_back(paramDeclAttr);
    moduleContext.parameterRefs[nttp->getNameAsString()] = paramRefAttr;
  }
}

} // namespace chwc::utils

#endif // CHWC_UTILS_TEMPLATE_H
