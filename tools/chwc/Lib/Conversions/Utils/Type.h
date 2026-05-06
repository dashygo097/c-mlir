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

  auto *recordDecl = recordType->getDecl();
  if (!recordDecl) {
    return nullptr;
  }

  return llvm::dyn_cast<clang::ClassTemplateSpecializationDecl>(recordDecl);
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
  if (!spec) {
    return std::nullopt;
  }

  if (getTemplateName(spec) != "UInt") {
    return std::nullopt;
  }

  const clang::TemplateArgumentList &args = spec->getTemplateArgs();
  if (args.size() != 1) {
    return std::nullopt;
  }

  const clang::TemplateArgument &widthArg = args[0];
  if (widthArg.getKind() != clang::TemplateArgument::Integral) {
    return std::nullopt;
  }

  return static_cast<unsigned>(widthArg.getAsIntegral().getZExtValue());
}

inline auto getSIntWidth(clang::QualType type) -> std::optional<unsigned> {
  const clang::ClassTemplateSpecializationDecl *spec = getTemplateSpec(type);
  if (!spec) {
    return std::nullopt;
  }

  if (getTemplateName(spec) != "SInt") {
    return std::nullopt;
  }

  const clang::TemplateArgumentList &args = spec->getTemplateArgs();
  if (args.size() != 1) {
    return std::nullopt;
  }

  const clang::TemplateArgument &widthArg = args[0];
  if (widthArg.getKind() != clang::TemplateArgument::Integral) {
    return std::nullopt;
  }

  return static_cast<unsigned>(widthArg.getAsIntegral().getZExtValue());
}

inline auto getIntTypeInfo(clang::QualType type) -> SignalTypeInfo {
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
  SignalTypeInfo valueInfo = getIntTypeInfo(type);
  if (valueInfo.isValue) {
    return valueInfo;
  }

  const clang::ClassTemplateSpecializationDecl *spec = getTemplateSpec(type);
  if (!spec) {
    return SignalTypeInfo{};
  }

  if (getTemplateName(spec) != "Signal") {
    return SignalTypeInfo{};
  }

  const clang::TemplateArgumentList &args = spec->getTemplateArgs();
  if (args.size() != 2) {
    return SignalTypeInfo{};
  }

  const clang::TemplateArgument &valueTypeArg = args[0];
  const clang::TemplateArgument &kindArg = args[1];

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
  info.fieldKind = *kind;
  info.width = elementInfo.width;
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
