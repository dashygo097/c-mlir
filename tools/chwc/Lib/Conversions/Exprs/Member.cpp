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

  HWFieldInfo &fieldInfo = fieldIt->second;

  switch (fieldInfo.kind) {
  case HWFieldKind::Input:
  case HWFieldKind::Wire:
  case HWFieldKind::Reg: {
    mlir::Value value = moduleContext.currentValues.lookup(fieldDecl);
    if (!value) {
      llvm::WithColor::error()
          << "chwc: hardware field is not wired yet: " << fieldInfo.name
          << "\n";
      return nullptr;
    }

    return value;
  }

  case HWFieldKind::Output: {
    mlir::Value value = moduleContext.outputValues.lookup(fieldDecl);
    if (!value) {
      llvm::WithColor::error()
          << "chwc: output field is read before assignment: " << fieldInfo.name
          << "\n";
      return nullptr;
    }

    return value;
  }
  }

  return nullptr;
}

} // namespace chwc
