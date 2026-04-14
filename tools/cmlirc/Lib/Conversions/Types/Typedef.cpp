#include "../../Converter.h"

namespace cmlirc {

auto CMLIRConverter::convertTypedefType(const clang::TypedefType *type)
    -> mlir::Type {
  return convertType(type->desugar());
}

} // namespace cmlirc
