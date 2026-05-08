#ifndef CHWC_UTILS_TYPE_H
#define CHWC_UTILS_TYPE_H

#include "../../Converter.h"
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
  unsigned width{0};
};

inline auto getTemplateSpec(clang::QualType type)
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

inline auto getTemplateName(const clang::ClassTemplateSpecializationDecl *spec)
    -> std::string {
  if (!spec || !spec->getSpecializedTemplate()) {
    return "";
  }

  return spec->getSpecializedTemplate()->getNameAsString();
}

inline auto getUIntWidth(clang::QualType type) -> std::optional<unsigned> {
  const clang::ClassTemplateSpecializationDecl *spec = getTemplateSpec(type);
  if (!spec || getTemplateName(spec) != "UInt") {
    return std::nullopt;
  }

  const clang::TemplateArgumentList &args = spec->getTemplateArgs();
  if (args.size() != 1 ||
      args[0].getKind() != clang::TemplateArgument::Integral) {
    return std::nullopt;
  }

  return static_cast<unsigned>(args[0].getAsIntegral().getZExtValue());
}

inline auto getSIntWidth(clang::QualType type) -> std::optional<unsigned> {
  const clang::ClassTemplateSpecializationDecl *spec = getTemplateSpec(type);
  if (!spec || getTemplateName(spec) != "SInt") {
    return std::nullopt;
  }

  const clang::TemplateArgumentList &args = spec->getTemplateArgs();
  if (args.size() != 1 ||
      args[0].getKind() != clang::TemplateArgument::Integral) {
    return std::nullopt;
  }

  return static_cast<unsigned>(args[0].getAsIntegral().getZExtValue());
}

inline auto getValueTypeInfo(clang::QualType type) -> SignalTypeInfo {
  SignalTypeInfo info;

  if (std::optional<unsigned> width = getUIntWidth(type)) {
    info.isValue = true;
    info.isSignal = false;
    info.isSigned = false;
    info.width = *width;
    return info;
  }

  if (std::optional<unsigned> width = getSIntWidth(type)) {
    info.isValue = true;
    info.isSignal = false;
    info.isSigned = true;
    info.width = *width;
    return info;
  }

  return info;
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
  SignalTypeInfo valueInfo = getValueTypeInfo(type);
  if (valueInfo.isValue) {
    return valueInfo;
  }

  const clang::ClassTemplateSpecializationDecl *spec = getTemplateSpec(type);
  if (!spec || getTemplateName(spec) != "Signal") {
    return SignalTypeInfo{};
  }

  const clang::TemplateArgumentList &args = spec->getTemplateArgs();
  if (args.size() != 2 || args[0].getKind() != clang::TemplateArgument::Type ||
      args[1].getKind() != clang::TemplateArgument::Integral) {
    return SignalTypeInfo{};
  }

  SignalTypeInfo elementInfo = getValueTypeInfo(args[0].getAsType());
  if (!elementInfo.isValue || elementInfo.isSignal) {
    return SignalTypeInfo{};
  }

  std::optional<HWFieldKind> kind =
      decodeObjectKind(args[1].getAsIntegral().getZExtValue());
  if (!kind) {
    return SignalTypeInfo{};
  }

  SignalTypeInfo info;
  info.isValue = true;
  info.isSignal = true;
  info.isSigned = elementInfo.isSigned;
  info.fieldKind = *kind;
  info.width = elementInfo.width;
  return info;
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

} // namespace chwc::utils

#endif // CHWC_UTILS_TYPE_H
