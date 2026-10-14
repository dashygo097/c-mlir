#include "../../Lowering.h"

namespace cmlir::frontend {

auto ASTLowering::generateAddrOfUnaryOperator(clang::Expr *expr)
    -> mlir::Value {
  // &*p does not access the pointed-to object, including when p is null.
  if (auto *unary =
          llvm::dyn_cast<clang::UnaryOperator>(expr->IgnoreParens())) {
    if (unary->getOpcode() == clang::UO_Deref) {
      return generateExpr(unary->getSubExpr());
    }
  }
  auto address = emitAddress(expr);
  if (mlir::failed(address)) {
    return {};
  }
  return address->materialize(context.builder(),
                              context.source().location(expr->getExprLoc()));
}

} // namespace cmlir::frontend
