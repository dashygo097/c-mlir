#include "../../Converter.h"
#include "../Utils/Constant.h"
#include "llvm/Support/WithColor.h"

namespace chwc {

auto CHWConverter::TraverseDeclStmt(clang::DeclStmt *declStmt) -> bool {
  mlir::OpBuilder &builder = contextManager.Builder();
  mlir::Location loc = builder.getUnknownLoc();

  for (clang::Decl *decl : declStmt->decls()) {
    auto *varDecl = mlir::dyn_cast<clang::VarDecl>(decl);
    if (!varDecl) {
      llvm::WithColor::error()
          << "chwc: only local var decl is supported in clock_tick()\n";
      continue;
    }

    mlir::Type type = convertType(varDecl->getType());
    if (!type) {
      continue;
    }

    mlir::Value value{};
    if (varDecl->hasInit()) {
      value = generateExpr(varDecl->getInit());
    } else {
      value = utils::zeroValue(builder, loc, type);
    }

    localValueTable[varDecl] = value;
  }

  return true;
}

} // namespace chwc
