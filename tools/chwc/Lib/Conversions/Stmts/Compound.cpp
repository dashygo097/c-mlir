#include "../../Converter.h"

namespace chwc {

auto CHWConverter::TraverseCompoundStmt(clang::CompoundStmt *compoundStmt)
    -> bool {
  if (!compoundStmt) {
    return true;
  }

  for (clang::Stmt *stmt : compoundStmt->body()) {
    if (!TraverseStmt(stmt)) {
      return false;
    }
  }

  return true;
}

} // namespace chwc
