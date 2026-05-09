#ifndef CHWC_UTILS_TYPE_H
#define CHWC_UTILS_TYPE_H

#include "../../Converter.h"
#include "clang/AST/DeclTemplate.h"
#include "clang/AST/Expr.h"
#include "clang/AST/Type.h"
#include <optional>
#include <string>

namespace chwc::utils {

struct SignalTypeInfo {
  bool isValue{false};
  bool isSignal{false};
  bool isSigned{false};
  std::optional<HWFieldKind> fieldKind;
  std::optional<unsigned> staticWidth;
  std::string parameterWidth;
};

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

inline auto decodeWidthArg(const clang::TemplateArgument &arg,
                           std::optional<unsigned> &staticWidth,
                           std::string &parameterWidth) -> bool {
  if (arg.getKind() == clang::TemplateArgument::Integral) {
    staticWidth = static_cast<unsigned>(arg.getAsIntegral().getZExtValue());
    parameterWidth.clear();
    return true;
  }

  if (arg.getKind() == clang::TemplateArgument::Expression) {
    std::optional<std::string> name =
        getTemplateWidthNameFromExpr(arg.getAsExpr());

    if (!name) {
      return false;
    }

    staticWidth.reset();
    parameterWidth = *name;
    return true;
  }

  if (arg.getKind() == clang::TemplateArgument::Declaration) {
    auto *decl =
        llvm::dyn_cast_or_null<clang::NonTypeTemplateParmDecl>(arg.getAsDecl());
    if (!decl) {
      return false;
    }

    staticWidth.reset();
    parameterWidth = decl->getNameAsString();
    return true;
  }

  return false;
}

inline auto decodeValueType(clang::QualType type, SignalTypeInfo &info)
    -> bool {
  std::string name = getTemplateSpecializationName(type);
  if (name != "UInt" && name != "SInt" && name != "Bool") {
    return false;
  }

  if (name == "Bool") {
    info.isValue = true;
    info.isSignal = false;
    info.isSigned = false;
    info.staticWidth = 1;
    info.parameterWidth.clear();
    return true;
  }

  if (getTemplateArgCount(type) != 1) {
    return false;
  }

  std::optional<clang::TemplateArgument> arg = getTemplateArg(type, 0);
  if (!arg) {
    return false;
  }

  std::optional<unsigned> staticWidth;
  std::string parameterWidth;

  if (!decodeWidthArg(*arg, staticWidth, parameterWidth)) {
    return false;
  }

  info.isValue = true;
  info.isSignal = false;
  info.isSigned = name == "SInt";
  info.staticWidth = staticWidth;
  info.parameterWidth = parameterWidth;
  return true;
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
  SignalTypeInfo info;

  if (decodeValueType(type, info)) {
    return info;
  }

  if (getTemplateSpecializationName(type) != "Signal") {
    return info;
  }

  if (getTemplateArgCount(type) != 2) {
    return info;
  }

  std::optional<clang::TemplateArgument> valueArg = getTemplateArg(type, 0);
  std::optional<clang::TemplateArgument> kindArg = getTemplateArg(type, 1);

  if (!valueArg || !kindArg) {
    return info;
  }

  if (valueArg->getKind() != clang::TemplateArgument::Type ||
      kindArg->getKind() != clang::TemplateArgument::Integral) {
    return info;
  }

  SignalTypeInfo elemInfo;
  if (!decodeValueType(valueArg->getAsType(), elemInfo)) {
    return info;
  }

  std::optional<HWFieldKind> kind =
      decodeObjectKind(kindArg->getAsIntegral().getZExtValue());

  if (!kind) {
    return info;
  }

  info = elemInfo;
  info.isSignal = true;
  info.fieldKind = *kind;
  return info;
}

inline auto getInstanceModuleDecl(clang::QualType type)
    -> const clang::CXXRecordDecl * {
  if (getTemplateSpecializationName(type) != "Instance") {
    return nullptr;
  }

  if (getTemplateArgCount(type) != 1) {
    return nullptr;
  }

  std::optional<clang::TemplateArgument> arg = getTemplateArg(type, 0);
  if (!arg || arg->getKind() != clang::TemplateArgument::Type) {
    return nullptr;
  }

  clang::QualType moduleType = arg->getAsType();
  moduleType = moduleType.getCanonicalType().getUnqualifiedType();

  return moduleType->getAsCXXRecordDecl();
}

inline auto getConstantArraySize(clang::QualType type)
    -> std::optional<uint64_t> {
  type = type.getCanonicalType().getUnqualifiedType();

  auto *arrayType =
      llvm::dyn_cast_or_null<clang::ConstantArrayType>(type.getTypePtr());

  if (!arrayType) {
    return std::nullopt;
  }

  return arrayType->getSize().getZExtValue();
}

inline auto getArrayElementType(clang::QualType type) -> clang::QualType {
  type = type.getCanonicalType().getUnqualifiedType();

  auto *arrayType = llvm::dyn_cast_or_null<clang::ArrayType>(type.getTypePtr());
  if (!arrayType) {
    return type;
  }

  return arrayType->getElementType();
}

inline auto getFieldElementType(clang::FieldDecl *fieldDecl)
    -> clang::QualType {
  if (!fieldDecl) {
    return clang::QualType{};
  }

  return getArrayElementType(fieldDecl->getType());
}

inline auto getFieldArraySize(clang::FieldDecl *fieldDecl)
    -> std::optional<uint64_t> {
  if (!fieldDecl) {
    return std::nullopt;
  }

  return getConstantArraySize(fieldDecl->getType());
}

inline auto getFieldElementTypeInfo(clang::FieldDecl *fieldDecl)
    -> SignalTypeInfo {
  if (!fieldDecl) {
    return SignalTypeInfo{};
  }

  return getSignalTypeInfo(getFieldElementType(fieldDecl));
}

inline auto isSignalType(clang::QualType type) -> bool {
  return getSignalTypeInfo(type).isSignal;
}

inline auto isValueType(clang::QualType type) -> bool {
  SignalTypeInfo info = getSignalTypeInfo(type);
  return info.isValue && !info.isSignal;
}

inline auto isSignedType(clang::QualType type) -> bool {
  return getSignalTypeInfo(type).isSigned;
}

inline auto isInstanceType(clang::QualType type) -> bool {
  return getInstanceModuleDecl(type) != nullptr;
}

} // namespace chwc::utils

#endif // CHWC_UTILS_TYPE_H
