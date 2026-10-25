#include "ExpressionLowering.h"
#include "llvm/Support/WithColor.h"

namespace cmlir::frontend {

auto ExpressionLowering::generateCXXThisExpr(clang::CXXThisExpr *thisExpr)
    -> mlir::Value {
  if (!function.thisValue) {
    llvm::WithColor::error()
        << "clang-to-mlir: 'this' used outside of an instance method\n";
    return nullptr;
  }
  return function.thisValue;
}

} // namespace cmlir::frontend
