#include "../../Converter.h"

namespace chwc {

auto CHWConverter::TraverseCompoundStmt(clang::CompoundStmt *compoundStmt)
    -> bool {
  for (clang::Stmt *stmt : compoundStmt->body()) {
    TraverseStmt(stmt);
  }

  return true;
}
} // namespace chwc
