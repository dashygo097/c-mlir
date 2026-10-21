#include "TypeLowering.h"

namespace cmlir::frontend {

auto TypeLowering::convertType(clang::QualType type) -> mlir::Type {
  const clang::Type *typePtr = type.getCanonicalType().getTypePtr();

#define REGISTER_TYPE(type)                                                    \
  if (auto *node = mlir::dyn_cast<clang::type>(typePtr)) {                     \
    auto result = convert##type(node);                                         \
    if (!result && !context.source().failed()) {                               \
      context.source().error({}, "type lowering failed");                      \
    }                                                                          \
    return result;                                                             \
  }

  REGISTER_TYPE(BuiltinType)
  REGISTER_TYPE(ArrayType)
  REGISTER_TYPE(PointerType)
  REGISTER_TYPE(TypedefType)
  REGISTER_TYPE(RecordType)
  REGISTER_TYPE(ReferenceType)

#undef REGISTER_TYPE

  context.source().error({}, "unsupported type: " + type.getAsString());
  return nullptr;
}

} // namespace cmlir::frontend
