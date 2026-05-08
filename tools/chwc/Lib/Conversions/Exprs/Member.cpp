#include "../../Converter.h"
#include "llvm/Support/WithColor.h"

namespace chwc {

auto CHWConverter::generateMemberExpr(clang::MemberExpr *memberExpr)
    -> mlir::Value {
  if (!memberExpr) {
    return nullptr;
  }

  auto *fieldDecl =
      mlir::dyn_cast<clang::FieldDecl>(memberExpr->getMemberDecl());
  if (!fieldDecl) {
    llvm::WithColor::error() << "chwc: unsupported member expr\n";
    return nullptr;
  }

  auto fieldIt = moduleContext.fields.find(fieldDecl);
  if (fieldIt == moduleContext.fields.end()) {
    llvm::WithColor::error()
        << "chwc: unknown hardware field: " << fieldDecl->getNameAsString()
        << "\n";
    return nullptr;
  }

  if (fieldIt->second.kind == HWFieldKind::Output) {
    llvm::WithColor::error() << "chwc: reading output field is not supported: "
                             << fieldIt->second.name << "\n";
    return nullptr;
  }

  mlir::Value value = moduleContext.currentValues.lookup(fieldDecl);
  if (!value) {
    llvm::WithColor::error()
        << "chwc: hardware field is not wired yet: " << fieldIt->second.name
        << "\n";
    return nullptr;
  }

  return value;
}

} // namespace chwc
