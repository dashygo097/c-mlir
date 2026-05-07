#include "../../Converter.h"
#include "../Utils/Constant.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/WithColor.h"

namespace chwc {

auto CHWConverter::generateDeclRefExpr(clang::DeclRefExpr *declRef)
    -> mlir::Value {
  if (!declRef) {
    return nullptr;
  }

  clang::ValueDecl *decl = declRef->getDecl();
  if (!decl) {
    return nullptr;
  }

  if (auto *templateParamDecl =
          llvm::dyn_cast<clang::NonTypeTemplateParmDecl>(decl)) {
    mlir::Value value = paramValueTable.lookup(templateParamDecl);
    if (value) {
      return value;
    }

    llvm::WithColor::error()
        << "chwc: template parameter value has not been emitted: "
        << templateParamDecl->getNameAsString() << "\n";
    return nullptr;
  }

  if (auto *fieldDecl = llvm::dyn_cast<clang::FieldDecl>(decl)) {
    return readFieldValue(fieldDecl);
  }

  if (auto *varDecl = llvm::dyn_cast<clang::VarDecl>(decl)) {
    mlir::Value value = localValueTable.lookup(varDecl);
    if (value) {
      return value;
    }

    llvm::WithColor::error()
        << "chwc: local variable has no value: " << varDecl->getNameAsString()
        << "\n";
    return nullptr;
  }

  if (auto *enumConstantDecl = llvm::dyn_cast<clang::EnumConstantDecl>(decl)) {
    mlir::OpBuilder &builder = contextManager.Builder();
    mlir::Location loc = builder.getUnknownLoc();

    mlir::Type type = convertType(enumConstantDecl->getType());
    if (!type) {
      return nullptr;
    }

    return utils::signedIntConst(builder, loc, type,
                                 enumConstantDecl->getInitVal().getSExtValue());
  }

  llvm::WithColor::error() << "chwc: unsupported DeclRefExpr: "
                           << decl->getNameAsString() << "\n";
  return nullptr;
}

} // namespace chwc
