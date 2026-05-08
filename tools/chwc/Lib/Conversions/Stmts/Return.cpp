#include "../../Converter.h"
#include "llvm/Support/WithColor.h"

namespace chwc {

auto CHWConverter::TraverseReturnStmt(clang::ReturnStmt *returnStmt) -> bool {
  if (functionStack.empty()) {
    llvm::WithColor::error() << "chwc: return outside function frame\n";
    return true;
  }

  clang::Expr *expr = returnStmt->getRetValue();
  if (!expr) {
    functionStack.back().returnValue = nullptr;
    functionStack.back().hasReturnValue = true;
    return true;
  }

  mlir::Value value = generateExpr(expr);
  if (!value) {
    return true;
  }

  functionStack.back().returnValue = value;
  functionStack.back().hasReturnValue = true;
  return true;
}

} // namespace chwc
