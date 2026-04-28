#ifndef CHWC_UTILS_ANNOTATION_H
#define CHWC_UTILS_ANNOTATION_H

#include "clang/AST/AST.h"

namespace chwc::utils {
auto getAnnotation(clang::Decl *decl) -> std::optional<std::string> {
  for (auto *attr : decl->specific_attrs<clang::AnnotateAttr>()) {
    return attr->getAnnotation().str();
  }

  return std::nullopt;
}

} // namespace chwc::utils

#endif // CHWC_UTILS_ANNOTATION_H
