#include "TypeLowering.h"

namespace cmlir::frontend {

auto TypeLowering::convertEnumType(const clang::EnumType *type) -> mlir::Type {
  auto underlying = type->getDecl()->getIntegerType();
  if (underlying.isNull()) {
    context.source().error(type->getDecl()->getLocation(),
                           "enum underlying type is unavailable");
    return {};
  }

  return convertType(underlying);
}

} // namespace cmlir::frontend
