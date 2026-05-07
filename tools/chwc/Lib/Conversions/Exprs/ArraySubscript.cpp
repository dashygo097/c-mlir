#include "../../Converter.h"
#include "../Utils/Expr.h"
#include "llvm/Support/WithColor.h"

namespace chwc {

auto CHWConverter::generateArraySubscriptExpr(
    clang::ArraySubscriptExpr *arraySub) -> mlir::Value {
  if (!arraySub) {
    return nullptr;
  }

  const clang::FieldDecl *fieldDecl =
      utils::getArrayBaseFieldDecl(arraySub->getBase());
  if (!fieldDecl) {
    llvm::WithColor::error()
        << "chwc: array read only supports hardware field arrays\n";
    return nullptr;
  }

  mlir::Value index = generateExpr(arraySub->getIdx());
  if (!index) {
    llvm::WithColor::error() << "chwc: failed to generate array index\n";
    return nullptr;
  }

  return readArrayElement(fieldDecl, index);
}

} // namespace chwc
