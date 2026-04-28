#include "../../Converter.h"
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

} // namespace chwc
