#include "../../Converter.h"
#include "llvm/Support/WithColor.h"

namespace chwc {

auto CHWConverter::generateMemberExpr(clang::MemberExpr *memberExpr)
    -> mlir::Value {
  auto *fieldDecl =
      mlir::dyn_cast<clang::FieldDecl>(memberExpr->getMemberDecl());

  if (!fieldDecl) {
    llvm::WithColor::error() << "chwc: member expr is not a field\n";
    return nullptr;
  }

  auto fieldIt = fieldTable.find(fieldDecl);
  if (fieldIt == fieldTable.end()) {
    llvm::WithColor::error()
        << "chwc: unknown hardware field: " << fieldDecl->getNameAsString()
        << "\n";
    return nullptr;
  }

  HWFieldInfo &fieldInfo = fieldIt->second;

  if (fieldInfo.kind == HWFieldKind::Output) {
    mlir::Value value = outputValueTable.lookup(fieldDecl);
    if (value) {
      return value;
    }
  }

  if (fieldInfo.kind == HWFieldKind::Reg ||
      fieldInfo.kind == HWFieldKind::Wire) {
    mlir::Value value = nextFieldValueTable.lookup(fieldDecl);
    if (value) {
      return value;
    }
  }

  mlir::Value value = currentFieldValueTable.lookup(fieldDecl);
  if (value) {
    return value;
  }

  llvm::WithColor::error() << "chwc: hardware field has no value: "
                           << fieldInfo.name << "\n";
  return nullptr;
}

} // namespace chwc
