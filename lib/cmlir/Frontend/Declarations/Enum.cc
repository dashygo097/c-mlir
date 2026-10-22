#include "../Lowering.h"

namespace cmlir::frontend {

auto ASTLowering::emitEnum(clang::EnumDecl *decl) -> bool {
  if (!decl) {
    return true;
  }

  return true;
}

} // namespace cmlir::frontend
