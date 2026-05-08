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

  if (constructExpr->getNumArgs() == 0) {
    mlir::Type type = convertType(constructExpr->getType());
    if (!type) {
      return nullptr;
    }

    mlir::OpBuilder &builder = contextManager.Builder();
    mlir::Location loc = builder.getUnknownLoc();

    return utils::zeroValue(builder, loc, type);
  }

  if (constructExpr->getNumArgs() == 1) {
    mlir::Value value = generateExpr(constructExpr->getArg(0));
    if (!value) {
      return nullptr;
    }

    mlir::Type type = convertType(constructExpr->getType());
    if (!type) {
      return value;
    }

    mlir::OpBuilder &builder = contextManager.Builder();
    mlir::Location loc = builder.getUnknownLoc();

    return utils::promoteValue(builder, loc, value, type);
  }

  llvm::WithColor::error() << "chwc: unsupported CXXConstructExpr with "
                           << constructExpr->getNumArgs() << " args\n";
  return nullptr;
}

} // namespace chwc
