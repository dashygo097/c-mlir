#include "../../Converter.h"

namespace chwc {

auto CHWConverter::convertBuiltinType(const clang::BuiltinType *type)
    -> mlir::Type {
  mlir::OpBuilder &builder = contextManager.Builder();

  switch (type->getKind()) {
  case clang::BuiltinType::Bool:
    return builder.getI1Type();

  case clang::BuiltinType::Char_U:
  case clang::BuiltinType::UChar:
  case clang::BuiltinType::Char_S:
  case clang::BuiltinType::SChar:
    return builder.getIntegerType(8);

  case clang::BuiltinType::UShort:
  case clang::BuiltinType::Short:
    return builder.getIntegerType(16);

  case clang::BuiltinType::UInt:
  case clang::BuiltinType::Int:
    return builder.getIntegerType(32);

  case clang::BuiltinType::ULong:
  case clang::BuiltinType::Long:
  case clang::BuiltinType::ULongLong:
  case clang::BuiltinType::LongLong:
    return builder.getIntegerType(64);

  case clang::BuiltinType::Void:
    return mlir::Type{};

  default:
    return nullptr;
  }
}

} // namespace chwc
