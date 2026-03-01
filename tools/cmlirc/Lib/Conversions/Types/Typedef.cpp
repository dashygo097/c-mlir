#include "../../Converter.h"

namespace cmlirc {
mlir::Type CMLIRConverter::convertTypedefType(const clang::TypedefType *type) {
  return convertType(type->desugar());
}
} // namespace cmlirc
