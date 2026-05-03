#ifndef CHWC_UTILS_TYPE_H
#define CHWC_UTILS_TYPE_H

#include "../../Converter.h"
#include "clang/AST/DeclTemplate.h"
#include "clang/AST/Type.h"
#include <optional>

namespace chwc::utils {

struct CHWCTypeInfo {
  bool isChwcType{false};
  bool isSignal{false};
  std::optional<HWFieldKind> fieldKind;
  unsigned width{0};
};

inline auto getTemplateName(const clang::ClassTemplateSpecializationDecl *spec)
    -> std::string {
  if (!spec || !spec->getSpecializedTemplate()) {
    return "";
  }

  return spec->getSpecializedTemplate()->getNameAsString();
}

inline auto getTemplateSpec(clang::QualType type)
    -> const clang::ClassTemplateSpecializationDecl * {
  type = type.getCanonicalType();

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

inline auto decodeObjectKind(uint64_t value) -> std::optional<HWFieldKind> {
  // Runtime enum:
  // Value = 0, Input = 1, Output = 2, Wire = 3, Reg = 4
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

inline auto getCHWCTypeInfo(clang::QualType type) -> CHWCTypeInfo {
  CHWCTypeInfo info;

  if (std::optional<unsigned> width = getUIntWidth(type)) {
    info.isChwcType = true;
    info.isSignal = false;
    info.width = *width;
    return info;
  }

  const clang::ClassTemplateSpecializationDecl *spec = getTemplateSpec(type);
  if (!spec) {
    return info;
  }

  if (getTemplateName(spec) != "Signal") {
    return info;
  }

  const clang::TemplateArgumentList &args = spec->getTemplateArgs();
  if (args.size() != 2) {
    return info;
  }

  const clang::TemplateArgument &valueTypeArg = args[0];
  const clang::TemplateArgument &kindArg = args[1];

  if (valueTypeArg.getKind() != clang::TemplateArgument::Type ||
      kindArg.getKind() != clang::TemplateArgument::Integral) {
    return info;
  }

  std::optional<unsigned> width = getUIntWidth(valueTypeArg.getAsType());
  if (!width) {
    return info;
  }

  std::optional<HWFieldKind> kind =
      decodeObjectKind(kindArg.getAsIntegral().getZExtValue());

  if (!kind) {
    return info;
  }

  info.isChwcType = true;
  info.isSignal = true;
  info.fieldKind = *kind;
  info.width = *width;
  return info;
}

inline auto isCHWCSignalType(clang::QualType type) -> bool {
  return getCHWCTypeInfo(type).isSignal;
}

} // namespace chwc::utils

#endif // CHWC_UTILS_TYPE_H
