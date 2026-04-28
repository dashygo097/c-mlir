#include "../../Converter.h"
#include "../Utils/HWOps.h"
#include "llvm/Support/WithColor.h"

namespace chwc {

auto CHWConverter::TraverseStmt(clang::Stmt *stmt) -> bool {
  if (!stmt) {
    return true;
  }

#define REGISTER_STMT_CONVERSION(StmtType)                                     \
  if (auto *node = mlir::dyn_cast<clang::StmtType>(stmt)) {                    \
    return Traverse##StmtType(node);                                           \
  }

  REGISTER_STMT_CONVERSION(CompoundStmt)
  REGISTER_STMT_CONVERSION(DeclStmt)
  REGISTER_STMT_CONVERSION(IfStmt)

#undef REGISTER_STMT_CONVERSION

  if (auto *expr = mlir::dyn_cast<clang::Expr>(stmt)) {
    generateExpr(expr);
    return true;
  }

  llvm::WithColor::error()
      << "chwc: unsupported statement conversion for stmt: "
      << stmt->getStmtClassName() << "\n";
  return true;
}

auto CHWConverter::TraverseCompoundStmt(clang::CompoundStmt *compoundStmt)
    -> bool {
  for (clang::Stmt *stmt : compoundStmt->body()) {
    TraverseStmt(stmt);
  }

  return true;
}

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
