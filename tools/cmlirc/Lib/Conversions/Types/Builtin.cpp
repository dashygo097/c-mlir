#include "../../Converter.h"
#include "llvm/Support/WithColor.h"

namespace cmlirc {

auto CMLIRConverter::convertBuiltinType(const clang::BuiltinType *type)
    -> mlir::Type {
  mlir::OpBuilder &builder = contextManager.Builder();

  switch (type->getKind()) {
  case clang::BuiltinType::Void:
    return builder.getNoneType();

  case clang::BuiltinType::Bool:
    return builder.getI1Type();

  case clang::BuiltinType::Char_S:
  case clang::BuiltinType::Char_U:
  case clang::BuiltinType::SChar:
  case clang::BuiltinType::UChar:
  case clang::BuiltinType::Char8:
    return builder.getI8Type();

  case clang::BuiltinType::Short:
  case clang::BuiltinType::UShort:
  case clang::BuiltinType::Char16:
    return builder.getI16Type();

  case clang::BuiltinType::Int:
  case clang::BuiltinType::UInt:
  case clang::BuiltinType::Char32:
  case clang::BuiltinType::WChar_S:
  case clang::BuiltinType::WChar_U:
    return builder.getI32Type();

  case clang::BuiltinType::Long:
  case clang::BuiltinType::ULong:
  case clang::BuiltinType::LongLong:
  case clang::BuiltinType::ULongLong:
  case clang::BuiltinType::Int128:
  case clang::BuiltinType::UInt128:
    return builder.getI64Type();

  case clang::BuiltinType::Half:
    return builder.getF16Type();

  case clang::BuiltinType::Float:
  case clang::BuiltinType::Float16:
    return builder.getF32Type();

  case clang::BuiltinType::Double:
  case clang::BuiltinType::LongDouble:
  case clang::BuiltinType::Float128:
    return builder.getF64Type();

  case clang::BuiltinType::NullPtr:
    llvm::WithColor::warning() << "cmlirc: nullptr_t mapped to i64\n";
    return builder.getI64Type();

  default:
    llvm::WithColor::error()
        << "cmlirc: unsupported builtin type: " << type << "\n";
  }

  return nullptr;
}
} // namespace cmlirc
