#include "../Lowering.h"

namespace cmlir::frontend {

auto ASTLowering::emitRecord(clang::RecordDecl *decl) -> bool {
  if (!decl || !decl->isCompleteDefinition()) {
    return true;
  }
  for (auto *child : decl->decls()) {
    if (!emitDeclaration(child)) {
      return false;
    }
  }
  return true;
}

} // namespace cmlir::frontend
