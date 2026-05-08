#include "../../Converter.h"
#include "../Utils/Cast.h"
#include "../Utils/Constant.h"
#include "llvm/Support/WithColor.h"

namespace chwc {

auto CHWConverter::generateCXXConstructExpr(
    clang::CXXConstructExpr *constructExpr) -> mlir::Value {
  if (!constructExpr) {
    return nullptr;
  }

  mlir::OpBuilder &builder = contextManager.Builder();
  mlir::Location loc = builder.getUnknownLoc();

  mlir::Type targetType = convertType(constructExpr->getType());
  if (!targetType) {
    return nullptr;
  }

  if (constructExpr->getNumArgs() == 0) {
    return utils::zeroValue(builder, loc, targetType);
  }

  if (constructExpr->getNumArgs() == 1) {
    clang::Expr *arg = constructExpr->getArg(0)->IgnoreParenImpCasts();

    if (auto *intLit = mlir::dyn_cast<clang::IntegerLiteral>(arg)) {
      return utils::intConst(builder, loc, targetType,
                             intLit->getValue().getSExtValue());
    }

    mlir::Value value = generateExpr(arg);
    if (!value) {
      return nullptr;
    }

    if (value.getType() == targetType) {
      return value;
    }

    return utils::promoteValue(builder, loc, value, targetType);
  }

  llvm::WithColor::error() << "chwc: unsupported CXXConstructExpr with "
                           << constructExpr->getNumArgs() << " args\n";
  return nullptr;
}

} // namespace chwc
