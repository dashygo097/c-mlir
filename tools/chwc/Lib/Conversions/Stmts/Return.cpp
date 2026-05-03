#include "../../Converter.h"
#include "llvm/Support/WithColor.h"

namespace chwc {

auto CHWConverter::TraverseReturnStmt(clang::ReturnStmt *returnStmt) -> bool {
  if (!returnStmt) {
    return true;
  }

  if (helperInlineDepth == 0) {
    llvm::WithColor::error()
        << "chwc: return statement is only supported inside helper method "
           "inlining\n";
    return false;
  }

  clang::Expr *retValue = returnStmt->getRetValue();
  if (!retValue) {
    currentReturnValue = nullptr;
    hasCurrentReturnValue = true;
    return false;
  }

  currentReturnValue = generateExpr(retValue);
  hasCurrentReturnValue = true;

  return false;
}

} // namespace chwc
