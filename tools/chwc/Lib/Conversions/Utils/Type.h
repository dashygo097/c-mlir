#ifndef CHWC_UTILS_TYPE_H
#define CHWC_UTILS_TYPE_H

#include "../../Converter.h"
#include "clang/AST/DeclTemplate.h"
#include "clang/AST/Expr.h"
#include "clang/AST/ExprCXX.h"
#include "clang/AST/Type.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Casting.h"
#include <cstdint>
#include <optional>
#include <string>

namespace chwc::utils {

struct SignalTypeInfo {
  bool isValue{false};
  bool isSignal{false};
  bool isSigned{false};
  bool isParametricWidth{false};

  std::optional<HWFieldKind> fieldKind;

  unsigned width{0};
  const clang::NonTypeTemplateParmDecl *widthParamDecl{nullptr};
};

struct ConstantArrayTypeInfo {
  bool isArray{false};
  clang::QualType elementType;
  uint64_t size{0};
  SignalTypeInfo elementInfo;
};

struct TemplateTypeInfo {
  bool valid{false};
  std::string name;
  llvm::SmallVector<clang::TemplateArgument, 4> args;
};

inline auto getTemplateNameString(clang::TemplateName templateName)
    -> std::string {
  clang::TemplateDecl *templateDecl = templateName.getAsTemplateDecl();
  if (!templateDecl) {
    return "";
  }

  return templateDecl->getNameAsString();
}

inline auto getTemplateTypeInfo(clang::QualType type) -> TemplateTypeInfo {
  TemplateTypeInfo info;

  type = type.getCanonicalType().getUnqualifiedType();

  const clang::Type *typePtr = type.getTypePtrOrNull();
  if (!typePtr) {
    return info;
  }

  if (auto *templateSpecType =
          llvm::dyn_cast<clang::TemplateSpecializationType>(typePtr)) {
    info.valid = true;
    info.name = getTemplateNameString(templateSpecType->getTemplateName());

    for (const clang::TemplateArgument &arg :
         templateSpecType->template_arguments()) {
      info.args.push_back(arg);
    }

    return info;
  }

  if (auto *recordType = typePtr->getAs<clang::RecordType>()) {
    auto *recordDecl = recordType->getDecl();
    auto *specDecl =
        llvm::dyn_cast_or_null<clang::ClassTemplateSpecializationDecl>(
            recordDecl);

    if (!specDecl || !specDecl->getSpecializedTemplate()) {
      return info;
    }

    info.valid = true;
    info.name = specDecl->getSpecializedTemplate()->getNameAsString();

    const clang::TemplateArgumentList &args = specDecl->getTemplateArgs();
    for (unsigned i = 0, e = args.size(); i < e; ++i) {
      info.args.push_back(args[i]);
    }

    return info;
  }

  return info;
}

inline auto getNonTypeTemplateParamFromExpr(clang::Expr *expr)
    -> const clang::NonTypeTemplateParmDecl * {
  if (!expr) {
    return nullptr;
  }

  expr = expr->IgnoreParenImpCasts();

  if (auto *constantExpr = llvm::dyn_cast_or_null<clang::ConstantExpr>(expr)) {
    return getNonTypeTemplateParamFromExpr(constantExpr->getSubExpr());
  }

  if (auto *implicitCast =
          llvm::dyn_cast_or_null<clang::ImplicitCastExpr>(expr)) {
    return getNonTypeTemplateParamFromExpr(implicitCast->getSubExpr());
  }

  if (auto *declRef = llvm::dyn_cast_or_null<clang::DeclRefExpr>(expr)) {
    return llvm::dyn_cast<clang::NonTypeTemplateParmDecl>(declRef->getDecl());
  }

  if (auto *subst =
          llvm::dyn_cast_or_null<clang::SubstNonTypeTemplateParmExpr>(expr)) {
    return llvm::dyn_cast_or_null<clang::NonTypeTemplateParmDecl>(
        subst->getParameter());
  }

  return nullptr;
}

inline auto getNonTypeTemplateParamFromArg(const clang::TemplateArgument &arg)
    -> const clang::NonTypeTemplateParmDecl * {
  switch (arg.getKind()) {
  case clang::TemplateArgument::Expression:
    return getNonTypeTemplateParamFromExpr(arg.getAsExpr());

  case clang::TemplateArgument::Declaration:
    return llvm::dyn_cast_or_null<clang::NonTypeTemplateParmDecl>(
        arg.getAsDecl());

  default:
    return nullptr;
  }
}

inline auto getIntTypeInfo(clang::QualType type) -> SignalTypeInfo {
  SignalTypeInfo info;

  TemplateTypeInfo templateInfo = getTemplateTypeInfo(type);
  if (!templateInfo.valid) {
    return info;
  }

  if (templateInfo.name != "UInt" && templateInfo.name != "SInt") {
    return info;
  }

  if (templateInfo.args.size() != 1) {
    return info;
  }

  const clang::TemplateArgument &widthArg = templateInfo.args[0];

  info.isValue = true;
  info.isSignal = false;
  info.isSigned = templateInfo.name == "SInt";

  if (widthArg.getKind() == clang::TemplateArgument::Integral) {
    info.width = static_cast<unsigned>(widthArg.getAsIntegral().getZExtValue());
    return info;
  }

  if (const clang::NonTypeTemplateParmDecl *widthParam =
          getNonTypeTemplateParamFromArg(widthArg)) {
    info.isParametricWidth = true;
    info.widthParamDecl = widthParam;
    return info;
  }

  return SignalTypeInfo{};
}

inline auto decodeObjectKind(uint64_t value) -> std::optional<HWFieldKind> {
  switch (value) {
  case 1:
    return HWFieldKind::Input;
  case 2:
    return HWFieldKind::Output;
  case 3:
    return HWFieldKind::Wire;
  case 4:
    return HWFieldKind::Reg;
  default:
    return std::nullopt;
  }
}

inline auto getSignalTypeInfo(clang::QualType type) -> SignalTypeInfo {
  SignalTypeInfo valueInfo = getIntTypeInfo(type);
  if (valueInfo.isValue) {
    return valueInfo;
  }

  TemplateTypeInfo templateInfo = getTemplateTypeInfo(type);
  if (!templateInfo.valid) {
    return SignalTypeInfo{};
  }

  if (templateInfo.name != "Signal") {
    return SignalTypeInfo{};
  }

  if (templateInfo.args.size() != 2) {
    return SignalTypeInfo{};
  }

  const clang::TemplateArgument &valueTypeArg = templateInfo.args[0];
  const clang::TemplateArgument &kindArg = templateInfo.args[1];

  if (valueTypeArg.getKind() != clang::TemplateArgument::Type ||
      kindArg.getKind() != clang::TemplateArgument::Integral) {
    return SignalTypeInfo{};
  }

  SignalTypeInfo elementInfo = getIntTypeInfo(valueTypeArg.getAsType());
  if (!elementInfo.isValue || elementInfo.isSignal) {
    return SignalTypeInfo{};
  }

  std::optional<HWFieldKind> kind =
      decodeObjectKind(kindArg.getAsIntegral().getZExtValue());
  if (!kind) {
    return SignalTypeInfo{};
  }

  SignalTypeInfo info;
  info.isValue = true;
  info.isSignal = true;
  info.isSigned = elementInfo.isSigned;
  info.isParametricWidth = elementInfo.isParametricWidth;
  info.width = elementInfo.width;
  info.widthParamDecl = elementInfo.widthParamDecl;
  info.fieldKind = *kind;
  return info;
}

inline auto getConstantArrayTypeInfo(clang::QualType type)
    -> ConstantArrayTypeInfo {
  ConstantArrayTypeInfo info;

  type = type.getCanonicalType().getUnqualifiedType();

  const clang::Type *typePtr = type.getTypePtrOrNull();
  if (!typePtr) {
    return info;
  }

  auto *arrayType = llvm::dyn_cast<clang::ConstantArrayType>(typePtr);
  if (!arrayType) {
    return info;
  }

  info.isArray = true;
  info.elementType = arrayType->getElementType();
  info.size = arrayType->getSize().getZExtValue();
  info.elementInfo = getSignalTypeInfo(info.elementType);
  return info;
}

inline auto isSignalType(clang::QualType type) -> bool {
  return getSignalTypeInfo(type).isSignal;
}

inline auto isUIntType(clang::QualType type) -> bool {
  SignalTypeInfo info = getSignalTypeInfo(type);
  return info.isValue && !info.isSignal && !info.isSigned;
}

inline auto isSIntType(clang::QualType type) -> bool {
  SignalTypeInfo info = getSignalTypeInfo(type);
  return info.isValue && !info.isSignal && info.isSigned;
}

} // namespace chwc::utils

#endif // CHWC_UTILS_TYPE_H
