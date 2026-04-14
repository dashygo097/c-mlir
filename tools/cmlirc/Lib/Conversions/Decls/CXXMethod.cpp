#include "../../Converter.h"

namespace cmlirc {

auto CMLIRConverter::TraverseCXXMethodDecl(clang::CXXMethodDecl *decl) -> bool {
  return TraverseFunctionDecl(decl);
}
} // namespace cmlirc
