#include "../../Converter.h"
#include "../Utils/Array.h"
#include "../Utils/Expr.h"
#include "llvm/Support/WithColor.h"

namespace chwc {

auto CHWConverter::generateArraySubscriptExpr(
    clang::ArraySubscriptExpr *arraySub) -> mlir::Value {
  if (!arraySub) {
    return nullptr;
  }

  clang::Expr *base = utils::ignoreCasts(arraySub->getBase());

  const clang::FieldDecl *fieldDecl = nullptr;

  if (auto *memberExpr = mlir::dyn_cast_or_null<clang::MemberExpr>(base)) {
    fieldDecl = mlir::dyn_cast<clang::FieldDecl>(memberExpr->getMemberDecl());
  } else if (auto *declRef = mlir::dyn_cast_or_null<clang::DeclRefExpr>(base)) {
    fieldDecl = mlir::dyn_cast<clang::FieldDecl>(declRef->getDecl());
  }

  if (!fieldDecl) {
    llvm::WithColor::error() << "chwc: unsupported array subscript base\n";
    return nullptr;
  }

  auto fieldIt = moduleContext.fields.find(fieldDecl);
  if (fieldIt == moduleContext.fields.end() || !fieldIt->second.isArray) {
    llvm::WithColor::error()
        << "chwc: array subscript base is not hardware array field\n";
    return nullptr;
  }

  mlir::Value arrayValue = moduleContext.currentValues.lookup(fieldDecl);
  if (!arrayValue) {
    llvm::WithColor::error()
        << "chwc: array field value is not available: " << fieldIt->second.name
        << "\n";
    return nullptr;
  }

  mlir::Value index = generateExpr(arraySub->getIdx());
  if (!index) {
    return nullptr;
  }

  mlir::OpBuilder &builder = contextManager.Builder();
  mlir::Location loc = builder.getUnknownLoc();

  return utils::arrayGet(builder, loc, arrayValue, index);
}

} // namespace chwc
