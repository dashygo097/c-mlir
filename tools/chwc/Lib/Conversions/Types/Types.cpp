#include "../../Converter.h"
#include "../Utils/Type.h"
#include "llvm/Support/WithColor.h"

namespace chwc {

auto CHWConverter::convertType(clang::QualType type) -> mlir::Type {
  mlir::OpBuilder &builder = contextManager.Builder();

  utils::SignalTypeInfo signalType = utils::getSignalTypeInfo(type);
  if (signalType.isValue) {
    return builder.getIntegerType(signalType.width);
  }

  const clang::Type *typePtr = type.getCanonicalType().getTypePtr();

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
