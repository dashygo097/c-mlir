#include "../../Converter.h"
#include "llvm/Support/WithColor.h"

namespace cmlirc {
auto CMLIRConverter::generateCXXThisExpr(clang::CXXThisExpr *thisExpr)
    -> mlir::Value {
  if (!currentThisValue) {
    llvm::WithColor::error()
        << "cmlirc: 'this' used outside of an instance method\n";
    return nullptr;
  }
  return currentThisValue;
}

} // namespace cmlirc
