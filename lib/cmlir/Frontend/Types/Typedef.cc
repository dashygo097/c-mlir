#include "TypeLowering.h"

namespace cmlir::frontend {

auto TypeLowering::convertTypedefType(const clang::TypedefType *type)
    -> mlir::Type {
  return convertType(type->desugar());
}

} // namespace cmlir::frontend
