#include "../../Converter.h"
#include "../Utils/Constant.h"

namespace chwc {

auto CHWConverter::TraverseDeclStmt(clang::DeclStmt *declStmt) -> bool {
  if (!declStmt) {
    return true;
  }

  if (functionStack.empty()) {
    functionStack.emplace_back();
  }

  for (clang::Decl *decl : declStmt->decls()) {
    auto *varDecl = mlir::dyn_cast<clang::VarDecl>(decl);
    if (!varDecl) {
      continue;
    }

    mlir::Value value = nullptr;

    if (varDecl->hasInit()) {
      value = generateExpr(varDecl->getInit());
    }

    if (!value) {
      mlir::Type type = convertType(varDecl->getType());
      if (!type) {
        continue;
      }

      mlir::OpBuilder &builder = contextManager.Builder();
      mlir::Location loc = builder.getUnknownLoc();

      value = utils::zeroValue(builder, loc, type);
    }

    functionStack.back().locals[varDecl] = value;
  }

  return true;
}

} // namespace chwc
