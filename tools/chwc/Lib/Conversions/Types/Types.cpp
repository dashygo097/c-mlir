#include "../../Converter.h"
#include "../Utils/Type.h"
#include "circt/Dialect/HW/HWTypes.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/WithColor.h"

namespace chwc {

auto CHWConverter::convertType(clang::QualType type) -> mlir::Type {
  mlir::OpBuilder &builder = contextManager.Builder();

  utils::ConstantArrayTypeInfo arrayInfo =
      utils::getConstantArrayTypeInfo(type);

  if (arrayInfo.isArray) {
    mlir::Type elementType = convertType(arrayInfo.elementType);
    if (!elementType) {
      llvm::WithColor::error()
          << "chwc: unsupported hardware array element type: "
          << arrayInfo.elementType.getAsString() << "\n";
      return nullptr;
    }

    return circt::hw::ArrayType::get(elementType, arrayInfo.size);
  }

  utils::SignalTypeInfo signalType = utils::getSignalTypeInfo(type);
  if (signalType.isValue) {
    return builder.getIntegerType(signalType.width);
  }

  const clang::Type *typePtr = type.getCanonicalType().getTypePtr();

#define REGISTER_TYPE(type)                                                    \
  if (auto *node = llvm::dyn_cast<clang::type>(typePtr)) {                     \
    return convert##type(llvm::cast<clang::type>(node));                       \
  }

  REGISTER_TYPE(BuiltinType)

#undef REGISTER_TYPE

  llvm::WithColor::error() << "chwc: unsupported type: " << type.getAsString()
                           << "\n";
  return nullptr;
}

} // namespace chwc
