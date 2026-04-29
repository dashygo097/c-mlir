#ifndef CHWC_UTILS_ANNOTATION_H
#define CHWC_UTILS_ANNOTATION_H

#include "clang/AST/Attr.h"
#include "clang/AST/DeclCXX.h"
#include "llvm/ADT/StringRef.h"
#include <optional>
#include <string>

namespace chwc::utils {

inline auto getAnnotation(clang::Decl *decl) -> std::optional<std::string> {
  if (!decl) {
    return std::nullopt;
  }

  for (auto *attr : decl->specific_attrs<clang::AnnotateAttr>()) {
    return attr->getAnnotation().str();
  }

  return std::nullopt;
}

inline auto hasAnnotation(clang::Decl *decl, llvm::StringRef expected) -> bool {
  if (!decl) {
    return false;
  }

  for (auto *attr : decl->specific_attrs<clang::AnnotateAttr>()) {
    if (attr->getAnnotation() == expected) {
      return true;
    }
  }

  return false;
}

inline auto isMethodNamed(clang::CXXMethodDecl *methodDecl,
                          llvm::StringRef name) -> bool {
  if (!methodDecl) {
    return false;
  }

  clang::DeclarationName declName = methodDecl->getDeclName();
  if (!declName.isIdentifier()) {
    return false;
  }

  clang::IdentifierInfo *identifier = declName.getAsIdentifierInfo();
  if (!identifier) {
    return false;
  }

  return identifier->getName() == name;
}

inline auto isResetMethod(clang::CXXMethodDecl *methodDecl) -> bool {
  if (!methodDecl) {
    return false;
  }

  if (hasAnnotation(methodDecl, "hw.reset")) {
    return true;
  }

  return isMethodNamed(methodDecl, "reset");
}

inline auto isClockTickMethod(clang::CXXMethodDecl *methodDecl) -> bool {
  if (!methodDecl) {
    return false;
  }

  if (hasAnnotation(methodDecl, "hw.clock_tick")) {
    return true;
  }

  return isMethodNamed(methodDecl, "clock_tick");
}

} // namespace chwc::utils

#endif // CHWC_UTILS_ANNOTATION_H
