#include "../../Converter.h"

namespace cmlirc {

mlir::Type CMLIRConverter::convertType(const clang::QualType type) {
  const clang::Type *typePtr = type.getCanonicalType().getTypePtr();

#define REGISTER_TYPE(type)                                                    \
  if (auto *node = mlir::dyn_cast<clang::type>(typePtr)) {                     \
    return convert##type(llvm::cast<clang::type>(node));                       \
  }

  REGISTER_TYPE(BuiltinType)
  REGISTER_TYPE(ArrayType)
  REGISTER_TYPE(PointerType)
  REGISTER_TYPE(TypedefType)
  REGISTER_TYPE(RecordType)

#undef REGISTER_TYPE

  llvm::errs() << "Unsupported type: " << type.getAsString();
  return nullptr;
}

} // namespace cmlirc
