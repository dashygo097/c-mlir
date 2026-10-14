#include "CallLowering.h"
#include "clang-mlir/Casts.h"
#include "clang-mlir/MemrefABI.h"
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
  auto source = value.getType();
  if (source == target) {
    return value;
  }
  if (auto memref = mlir::dyn_cast<mlir::MemRefType>(target)) {
    if (auto result =
            clang_mlir::coerceMemRefForCall(builder, loc, value, memref)) {
      return result;
    }
  } else if (mlir::isa<mlir::LLVM::LLVMPointerType>(target)) {
    if (mlir::isa<mlir::MemRefType>(source)) {
      return clang_mlir::memrefToPointer(builder, loc, value, target);
    }
    if (source.isIndex() || mlir::isa<mlir::IntegerType>(source)) {
      return clang_mlir::integerToPointer(builder, loc, value, target);
    }
  } else if (target.isIntOrFloat() &&
             (source.isIndex() || source.isIntOrFloat())) {
    return clang_mlir::toValue(
        builder, loc, value, target,
        !argument.clangType->isUnsignedIntegerOrEnumerationType());
  }
  mlir::emitError(loc) << "cannot convert function argument: " << source
                       << " -> " << target;
  return {};
}

} // namespace cmlir::frontend
