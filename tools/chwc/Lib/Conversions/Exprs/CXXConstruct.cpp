#include "../../Converter.h"
#include "../Utils/Cast.h"
#include "../Utils/Constant.h"
#include "llvm/Support/WithColor.h"

namespace chwc {

auto CHWConverter::generateCXXConstructExpr(clang::CXXConstructExpr *expr)
    -> mlir::Value {
  if (!expr) {
    return nullptr;
  }

  mlir::OpBuilder &builder = contextManager.Builder();
  mlir::Location loc = builder.getUnknownLoc();

  mlir::Type targetType = convertType(expr->getType());
  if (!targetType) {
    return nullptr;
  }

  if (expr->getNumArgs() == 0) {
    return utils::zeroValue(builder, loc, targetType);
  }

  if (expr->getNumArgs() == 1) {
    mlir::Value value = generateExpr(expr->getArg(0));
    if (!value) {
      return nullptr;
    }

    return utils::promoteValue(builder, loc, value, targetType);
  }

  llvm::WithColor::error() << "chwc: unsupported CXXConstructExpr with "
                           << expr->getNumArgs() << " args\n";
  return nullptr;
}

} // namespace chwc
