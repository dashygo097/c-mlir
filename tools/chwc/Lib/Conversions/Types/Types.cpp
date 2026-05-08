#include "../../Converter.h"
#include "../Utils/Type.h"
#include "circt/Dialect/HW/HWTypes.h"
#include "llvm/Support/WithColor.h"

namespace chwc {

auto CHWConverter::convertType(clang::QualType type) -> mlir::Type {
  mlir::OpBuilder &builder = contextManager.Builder();

  type = type.getCanonicalType().getUnqualifiedType();

  if (std::optional<uint64_t> size = utils::getConstantArraySize(type)) {
    mlir::Type elementType = convertType(utils::getArrayElementType(type));
    if (!elementType) {
      return nullptr;
    }

    return circt::hw::ArrayType::get(elementType, *size);
  }

  utils::SignalTypeInfo typeInfo = utils::getSignalTypeInfo(type);
  if (typeInfo.isValue) {
    return builder.getIntegerType(typeInfo.width);
  }

  const clang::Type *typePtr = type.getTypePtr();

#define REGISTER_TYPE(type)                                                    \
  if (auto *node = mlir::dyn_cast<clang::type>(typePtr)) {                     \
    return convert##type(mlir::cast<clang::type>(node));                       \
  }

  REGISTER_TYPE(BuiltinType)

#undef REGISTER_TYPE

  llvm::WithColor::error() << "chwc: unsupported type: " << type.getAsString()
                           << "\n";
  return nullptr;
}

} // namespace chwc
