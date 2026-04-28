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

  mlir::Value value = currentFieldValueTable.lookup(fieldDecl);
  if (value) {
    return value;
  }

  value = outputValueTable.lookup(fieldDecl);
  if (value) {
    return value;
  }

  llvm::WithColor::error() << "chwc: unknown hardware field: "
                           << fieldDecl->getNameAsString() << "\n";
  return nullptr;
}

} // namespace chwc
