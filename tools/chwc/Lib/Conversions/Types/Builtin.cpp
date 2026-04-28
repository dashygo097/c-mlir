#include "../../Converter.h"
#include "llvm/Support/WithColor.h"

namespace chwc {

auto CHWConverter::convertBuiltinType(const clang::BuiltinType *type)
    -> mlir::Type {
  mlir::OpBuilder &builder = contextManager.Builder();

  switch (type->getKind()) {
  case clang::BuiltinType::Bool:
    return builder.getI1Type();

  case clang::BuiltinType::Char_U:
  case clang::BuiltinType::Char_S:
  case clang::BuiltinType::UChar:
  case clang::BuiltinType::SChar:
    return builder.getI8Type();

  case clang::BuiltinType::UShort:
  case clang::BuiltinType::Short:
    return builder.getI16Type();

  case clang::BuiltinType::UInt:
  case clang::BuiltinType::Int:
    return builder.getI32Type();

  case clang::BuiltinType::ULong:
  case clang::BuiltinType::Long:
  case clang::BuiltinType::ULongLong:
  case clang::BuiltinType::LongLong:
    return builder.getI64Type();

  default:
    llvm::WithColor::error() << "chwc: unsupported builtin type\n";
    return nullptr;
  }
}

} // namespace chwc
