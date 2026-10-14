#include "ExternalABI.h"
#include "clang-mlir/Casts.h"
#include "mlir/IR/Diagnostics.h"

namespace cmlir::frontend {

namespace {

auto promoteVariadicArgument(mlir::OpBuilder &builder, mlir::Location loc,
                             const CallArg &argument) -> mlir::Value {
  auto value = argument.value;
  auto type = value.getType();
  if (mlir::isa<mlir::MemRefType>(type)) {
    return clang_mlir::memrefToPointer(
        builder, loc, value,
        mlir::LLVM::LLVMPointerType::get(builder.getContext()));
  }
  if (type.isIndex()) {
    return clang_mlir::indexToInteger(builder, loc, value,
                                      builder.getI64Type());
  }
  if (auto integer = mlir::dyn_cast<mlir::IntegerType>(type)) {
    if (integer.getWidth() < 32) {
      return clang_mlir::toValue(
          builder, loc, value, builder.getI32Type(),
          !argument.clangType->isUnsignedIntegerOrEnumerationType());
    }
    return value;
  }
  if (auto floating = mlir::dyn_cast<mlir::FloatType>(type)) {
    if (floating.getWidth() < 64) {
      return clang_mlir::toValue(builder, loc, value, builder.getF64Type());
    }
    return value;
  }
  if (mlir::isa<mlir::LLVM::LLVMPointerType>(type)) {
    return value;
  }
  mlir::emitError(loc) << "unsupported variadic C ABI argument type: " << type;
  return {};
}

} // namespace

auto emitVariadicCall(mlir::OpBuilder &builder, mlir::Location loc,
                      mlir::ModuleOp module, llvm::StringRef name,
                      llvm::ArrayRef<CallArg> arguments,
                      mlir::FunctionType signature)
    -> mlir::FailureOr<mlir::Value> {
  llvm::SmallVector<mlir::Value, 8> prepared;
  for (auto [index, argument] : llvm::enumerate(arguments)) {
    auto value = index < signature.getNumInputs()
                     ? prepareCallArgument(builder, loc, argument,
                                           signature.getInput(index))
                     : promoteVariadicArgument(builder, loc, argument);
    if (!value) {
      return mlir::failure();
    }
    prepared.push_back(value);
  }
  auto resultType =
      signature.getNumResults() == 0
          ? mlir::Type(mlir::LLVM::LLVMVoidType::get(builder.getContext()))
          : signature.getResult(0);
  auto functionType = mlir::LLVM::LLVMFunctionType::get(
      resultType, signature.getInputs(), true);
  if (!getOrCreateLLVMFunctionDecl(builder, module, name, functionType)) {
    return mlir::failure();
  }
  auto call = mlir::LLVM::CallOp::create(
      builder, loc, signature.getResults(),
      mlir::FlatSymbolRefAttr::get(builder.getContext(), name), prepared);
  call.setVarCalleeType(functionType);
  return call.getNumResults() == 0 ? mlir::Value{} : call.getResult();
}

} // namespace cmlir::frontend
