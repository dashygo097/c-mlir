#include "../Lowering.h"
#include "clang/AST/DeclCXX.h"
#include "clang/Basic/SourceManager.h"

namespace cmlir::frontend {

auto ASTLowering::emitDeclaration(clang::Decl *decl) -> bool {
  if (!decl || decl->isImplicit()) {
    return true;
  }
  if (context.ast().getSourceManager().isInSystemHeader(decl->getLocation())) {
    return true;
  }
  if (auto *fn = llvm::dyn_cast<clang::FunctionDecl>(decl)) {
    return emitFunction(fn);
  }
  if (auto *record = llvm::dyn_cast<clang::RecordDecl>(decl)) {
    return emitRecord(record);
  }
  if (auto *_enum = llvm::dyn_cast<clang::EnumDecl>(decl)) {
    return emitEnum(_enum);
  }
  if (auto *var = llvm::dyn_cast<clang::VarDecl>(decl)) {
    return emitVariable(var);
  }
  if (auto *context = llvm::dyn_cast<clang::DeclContext>(decl)) {
    for (auto *child : context->decls()) {
      if (!emitDeclaration(child)) {
        return false;
      }
    }
    return true;
  }
  if (llvm::isa<clang::TypedefNameDecl, clang::EmptyDecl, clang::AccessSpecDecl,
                clang::FieldDecl>(decl)) {
    return true;
  }
  context.source().error(decl->getLocation(), "unsupported declaration");
  return false;
}

} // namespace cmlir::frontend
