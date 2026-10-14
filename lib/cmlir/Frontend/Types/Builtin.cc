#include "TypeLowering.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "llvm/ADT/APFloat.h"

namespace cmlir::frontend {

auto TypeLowering::convertBuiltinType(const clang::BuiltinType *type)
    -> mlir::Type {
  auto &builder = context.builder();
  auto &ast = context.ast();
  auto source = clang::QualType(type, 0);
  if (type->getKind() == clang::BuiltinType::Void) {
    return builder.getNoneType();
  }
  if (type->getKind() == clang::BuiltinType::Bool) {
    return builder.getI1Type();
  }
  if (type->isInteger()) {
    return builder.getIntegerType(ast.getTypeSize(source));
  }
  if (type->isFloatingPoint()) {
    auto &semantics = ast.getFloatTypeSemantics(source);
    if (&semantics == &llvm::APFloat::BFloat()) {
      return builder.getBF16Type();
    }
    switch (llvm::APFloat::semanticsSizeInBits(semantics)) {
    case 16:
      return builder.getF16Type();
    case 32:
      return builder.getF32Type();
    case 64:
      return builder.getF64Type();
    case 80:
      return builder.getF80Type();
    case 128:
      return builder.getF128Type();
    default:
      break;
    }
  }
  if (type->getKind() == clang::BuiltinType::NullPtr) {
    return mlir::LLVM::LLVMPointerType::get(builder.getContext());
  }
  context.source().error({}, "unsupported builtin type");
  return {};
}

} // namespace cmlir::frontend
