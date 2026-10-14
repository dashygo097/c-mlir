#include "CallLowering.h"
#include "../Types/TypeLowering.h"
#include "ExternalABI.h"
#include "Math.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/Diagnostics.h"

namespace cmlir::frontend {

auto CallLowering::emit(const clang::FunctionDecl &callee,
                        clang::SourceLocation sourceLoc,
                        llvm::ArrayRef<CallArg> arguments)
    -> mlir::FailureOr<mlir::Value> {
  auto convention = callConvention(callee);
  bool variadic = callee.isVariadic();
  if (callee.isVariadic() && convention == CallConvention::Internal) {
    context.source().error(sourceLoc,
                           "variadic function definitions are unsupported");
    return mlir::failure();
  }
  auto signature = lowerFunctionSignature(types, callee, convention);
  if (mlir::failed(signature)) {
    context.source().error(sourceLoc, "unsupported function call signature");
    return mlir::failure();
  }
  // Clang's type-generic builtins (e.g. __builtin_isfinite) have no
  // prototype. Only recognized intrinsics may specialize that signature
  // from their already evaluated operands.
  if (callee.getBuiltinID() && (callee.getType()->isFunctionNoProtoType() ||
                                (variadic && callee.getNumParams() == 0))) {
    llvm::SmallVector<mlir::Type, 4> inputs;
    for (const auto &argument : arguments) {
      inputs.push_back(argument.value.getType());
    }
    auto specialized = mlir::FunctionType::get(signature->getContext(), inputs,
                                               signature->getResults());
    if (!findMathIntrinsic(callee, specialized)) {
      context.source().error(sourceLoc, "unsupported compiler builtin call");
      return mlir::failure();
    }
    signature = specialized;
    variadic = false;
  }
  auto &builder = context.builder();
  auto loc = context.source().location(sourceLoc);
  auto name = functionSymbolName(callee);
  auto fixedCount = signature->getNumInputs();
  if (arguments.size() < fixedCount ||
      (!variadic && arguments.size() != fixedCount)) {
    mlir::emitError(loc) << "function '" << name << "' expects "
                         << (variadic ? "at least " : "") << fixedCount
                         << " argument(s), got " << arguments.size();
    return mlir::failure();
  }
  if (variadic) {
    return emitVariadicCall(builder, loc, context.module(), name, arguments,
                            *signature);
  }

  llvm::SmallVector<mlir::Value, 8> prepared;
  for (auto [argument, type] : llvm::zip(arguments, signature->getInputs())) {
    auto value = prepareCallArgument(builder, loc, argument, type);
    if (!value) {
      return mlir::failure();
    }
    prepared.push_back(value);
  }
  if (convention == CallConvention::External) {
    if (const auto *intrinsic = findMathIntrinsic(callee, *signature)) {
      auto value = intrinsic->emit(builder, loc, prepared);
      if (value.getType() != signature->getResult(0)) {
        // Floating-point classification returns i1 in MLIR and int in C.
        value = mlir::arith::ExtUIOp::create(builder, loc,
                                             signature->getResult(0), value);
      }
      return value;
    }
    if (const auto *identifier = callee.getIdentifier();
        identifier && identifier->getName().starts_with("__builtin_")) {
      context.source().error(sourceLoc, "unsupported compiler builtin call");
      return mlir::failure();
    }
  }
  auto function =
      getOrCreateFunctionDecl(builder, context.module(), name, *signature);
  if (!function) {
    return mlir::failure();
  }
  auto call = mlir::func::CallOp::create(builder, loc, function, prepared);
  return call.getNumResults() == 0 ? mlir::Value{} : call.getResult(0);
}

} // namespace cmlir::frontend
