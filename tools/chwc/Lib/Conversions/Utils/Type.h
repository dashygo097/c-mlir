#ifndef CHWC_UTILS_TYPE_H
#define CHWC_UTILS_TYPE_H

#include "../../Converter.h"
#include "./Template.h"
#include "clang/AST/DeclTemplate.h"
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

  if (name == "Bool") {
    info.isValue = true;
    info.isSignal = false;
    info.isSigned = false;
    info.staticWidth = 1;
    info.parameterWidth.clear();
    return true;
  }

  if (name == "UInt") {
    if (getTemplateArgCount(type) != 1) {
      return false;
    }

    std::optional<clang::TemplateArgument> widthArg = getTemplateArg(type, 0);
    if (!widthArg) {
      return false;
    }

    std::optional<unsigned> staticWidth;
    std::string parameterWidth;

    if (!decodeWidthArg(*widthArg, staticWidth, parameterWidth)) {
      return false;
    }

    info.isValue = true;
    info.isSignal = false;
    info.isSigned = false;
    info.staticWidth = staticWidth;
    info.parameterWidth = parameterWidth;
    return true;
  }

  if (name == "SInt") {
    if (getTemplateArgCount(type) != 1) {
      return false;
    }

    std::optional<clang::TemplateArgument> widthArg = getTemplateArg(type, 0);
    if (!widthArg) {
      return false;
    }

    std::optional<unsigned> staticWidth;
    std::string parameterWidth;

    if (!decodeWidthArg(*widthArg, staticWidth, parameterWidth)) {
      return false;
    }

    info.isValue = true;
    info.isSignal = false;
    info.isSigned = true;
    info.staticWidth = staticWidth;
    info.parameterWidth = parameterWidth;
    return true;
  }

  if (name == "Enum") {
    if (getTemplateArgCount(type) != 1) {
      return false;
    }

    std::optional<clang::TemplateArgument> numberArg = getTemplateArg(type, 0);
    if (!numberArg) {
      return false;
    }

    if (numberArg->getKind() != clang::TemplateArgument::Integral) {
      return false;
    }

    uint64_t number = numberArg->getAsIntegral().getZExtValue();

    unsigned width = 0;
    uint64_t values = 1;

    while (values < number) {
      values <<= 1;
      ++width;
    }

    if (width == 0) {
      width = 1;
    }

    info.isValue = true;
    info.isSignal = false;
    info.isSigned = false;
    info.staticWidth = width;
    info.parameterWidth.clear();
    return true;
  }

  return false;
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

inline auto stripArrayElementType(clang::QualType type) -> clang::QualType {
  type = type.getCanonicalType().getUnqualifiedType();

  auto *arrayType = llvm::dyn_cast_or_null<clang::ArrayType>(type.getTypePtr());
  if (!arrayType) {
    return type;
  }

  return arrayType->getElementType();
}

inline auto getSignalTypeInfo(clang::QualType type) -> SignalTypeInfo {
  SignalTypeInfo info;

  if (decodeValueType(type, info)) {
    return info;
  }

  if (getTemplateSpecializationName(type) == "RegInit") {
    if (getTemplateArgCount(type) != 2) {
      return info;
    }

    std::optional<clang::TemplateArgument> valueArg = getTemplateArg(type, 0);
    std::optional<clang::TemplateArgument> initArg = getTemplateArg(type, 1);

    if (!valueArg || !initArg) {
      return info;
    }

    if (valueArg->getKind() != clang::TemplateArgument::Type ||
        initArg->getKind() != clang::TemplateArgument::Integral) {
      return info;
    }

    SignalTypeInfo valueInfo;
    if (!decodeValueType(valueArg->getAsType(), valueInfo)) {
      return info;
    }

    info = valueInfo;
    info.isSignal = true;
    info.fieldKind = HWFieldKind::Reg;
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

inline auto getRegInitValue(clang::QualType type) -> std::optional<int64_t> {
  type = stripArrayElementType(type);

  if (getTemplateSpecializationName(type) != "RegInit") {
    return std::nullopt;
  }

  if (getTemplateArgCount(type) != 2) {
    return std::nullopt;
  }

  std::optional<clang::TemplateArgument> initArg = getTemplateArg(type, 1);
  if (!initArg) {
    return std::nullopt;
  }

  if (initArg->getKind() != clang::TemplateArgument::Integral) {
    return std::nullopt;
  }

  return initArg->getAsIntegral().getSExtValue();
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
