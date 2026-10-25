#include "CallLowering.h"
#include "clang-mlir/Casts.h"
#include "mlir/IR/Diagnostics.h"

namespace cmlir::frontend {

auto prepareCallArgument(mlir::OpBuilder &builder, mlir::Location loc,
                         const CallArg &argument, mlir::Type target)
    -> mlir::Value {
  auto value = argument.value;
  if (!value) {
    mlir::emitError(loc, "missing function call argument");
    return {};
  }
  if (auto memref = mlir::dyn_cast<mlir::MemRefType>(value.getType());
      memref && mlir::isa<mlir::LLVM::LLVMPointerType>(target) &&
      mlir::isa<mlir::BaseMemRefType>(memref.getElementType())) {
    mlir::emitError(
        loc,
        "native pointer arguments cannot address memref descriptor storage");
    return {};
  }
  if (auto converted = clang_mlir::toValue(
          builder, loc, value, target,
          !argument.clangType->isUnsignedIntegerOrEnumerationType())) {
    return converted;
  }
  mlir::emitError(loc) << "cannot convert function argument: "
                       << value.getType() << " -> " << target;
  return {};
}

} // namespace cmlir::frontend
