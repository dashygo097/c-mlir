#include "../../Converter.h"
#include "../Utils/Cast.h"

namespace chwc {

auto CHWConverter::TraverseDeclStmt(clang::DeclStmt *declStmt) -> bool {
  if (!declStmt) {
    return true;
  }

  for (clang::Decl *decl : declStmt->decls()) {
    auto *varDecl = llvm::dyn_cast_or_null<clang::VarDecl>(decl);
    if (!varDecl) {
      continue;
    }

    if (!varDecl->hasInit()) {
      continue;
    }

    mlir::Value initValue = generateExpr(varDecl->getInit());
    if (!initValue) {
      llvm::WithColor::error() << "chwc: failed to generate local initializer: "
                               << varDecl->getNameAsString() << "\n";
      continue;
    }

    clang::QualType varType = varDecl->getType();

    if (!varType->isUndeducedAutoType() && !varType->getContainedAutoType()) {
      mlir::Type targetType = convertType(varType);

      if (targetType && initValue.getType() != targetType) {
        mlir::OpBuilder &builder = contextManager.Builder();
        mlir::Location loc = builder.getUnknownLoc();

        initValue = utils::promoteValue(builder, loc, initValue, targetType);
        if (!initValue) {
          continue;
        }
      }
    }

    if (functionStack.empty()) {
      functionStack.emplace_back();
    }

    functionStack.back().locals[varDecl] = initValue;
  }

  return true;
}

} // namespace chwc
