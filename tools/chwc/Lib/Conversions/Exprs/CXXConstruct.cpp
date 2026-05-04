#include "../../Converter.h"
#include "../Utils/Constant.h"
#include "llvm/Support/WithColor.h"

namespace chwc {

auto CHWConverter::generateCXXConstructExpr(clang::CXXConstructExpr *expr)
    -> mlir::Value {
  if (!expr) {
    return nullptr;
  }

  if (expr->getNumArgs() == 0) {
    mlir::OpBuilder &builder = contextManager.Builder();
    mlir::Location loc = builder.getUnknownLoc();

    mlir::Type type = convertType(expr->getType());
    if (!type) {
      return nullptr;
    }

    return utils::zeroValue(builder, loc, type);
  }

  if (expr->getNumArgs() == 1) {
    return generateExpr(expr->getArg(0));
  }

  llvm::WithColor::error() << "chwc: unsupported CXXConstructExpr with "
                           << expr->getNumArgs() << " args\n";
  return nullptr;
}

} // namespace chwc
