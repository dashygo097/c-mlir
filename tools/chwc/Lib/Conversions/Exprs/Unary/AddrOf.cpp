#include "../../../Converter.h"

#include "llvm/Support/WithColor.h"

namespace chwc {

auto CHWConverter::generateAddrOfUnaryOperator(clang::Expr *addrOfExpr)
    -> mlir::Value {
  (void)addrOfExpr;

  llvm::WithColor::error()
      << "chwc: address-of operator is unsupported in hardware DSL\n";
  return nullptr;
}

} // namespace chwc
