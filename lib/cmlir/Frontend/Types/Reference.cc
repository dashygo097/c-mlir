#include "TypeLowering.h"

namespace cmlir::frontend {

auto TypeLowering::convertReferenceType(const clang::ReferenceType *)
    -> mlir::Type {
  context.source().error({}, "reference bindings are unsupported");
  return {};
}

} // namespace cmlir::frontend
