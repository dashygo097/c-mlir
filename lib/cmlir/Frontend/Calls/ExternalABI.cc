#include "../Lowering.h"
#include "CallLowering.h"
#include "clang-mlir/Casts.h"
#include "clang-mlir/Constants.h"
#include "clang-mlir/MemrefABI.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/WithColor.h"

#include "mlir/IR/Diagnostics.h"

namespace cmlir::frontend {

static auto isUnsignedIntegerLike(clang::QualType type) -> bool {
  if (type.isNull()) {
    return false;
  }

  type = type.getCanonicalType();

  if (type->isBooleanType()) {
    return true;
  }

  return type->isUnsignedIntegerType();
}

static auto isScalarLLVMCompatible(mlir::Type type) -> bool {
  return mlir::isa<mlir::IntegerType>(type) ||
         mlir::isa<mlir::FloatType>(type) ||
         mlir::isa<mlir::LLVM::LLVMPointerType>(type);
}

static auto materializeLlvmPointer(mlir::OpBuilder &builder, mlir::Location loc,
                                   mlir::Value value) -> mlir::Value {
  mlir::Type type = value.getType();
  auto ptrType = mlir::LLVM::LLVMPointerType::get(builder.getContext());

  if (mlir::isa<mlir::LLVM::LLVMPointerType>(type)) {
    return value;
  }

  if (mlir::isa<mlir::MemRefType>(type)) {
    return clang_mlir::memrefToPointer(
        builder, loc, value,
        mlir::LLVM::LLVMPointerType::get(builder.getContext()));
  }

  if (type.isIndex() || mlir::isa<mlir::IntegerType>(type)) {
    return clang_mlir::integerToPointer(builder, loc, value, ptrType);
  }

  mlir::emitError(loc)
      << "clang-to-mlir: cannot materialize LLVM pointer from value of type "
      << type << "\n";
  return nullptr;
}

static auto coerceExternalArg(mlir::OpBuilder &builder, mlir::Location loc,
                              const CallArg &arg, mlir::Type targetType)
    -> mlir::Value {
  mlir::Value value = arg.value;
  if (!value) {
    return {};
  }
  mlir::Type srcType = value.getType();

  if (srcType == targetType) {
    return value;
  }

  if (mlir::isa<mlir::LLVM::LLVMPointerType>(targetType)) {
    return materializeLlvmPointer(builder, loc, value);
  }

  if (mlir::isa<mlir::MemRefType>(srcType)) {
    mlir::emitError(loc) << "clang-to-mlir: cannot pass memref value to "
                            "non-pointer C ABI argument: "
                         << srcType << " -> " << targetType << "\n";
    return nullptr;
  }

  if (auto targetInt = mlir::dyn_cast<mlir::IntegerType>(targetType)) {
    if (srcType.isIndex()) {
      return clang_mlir::indexToInteger(builder, loc, value, targetInt);
    }

    if (mlir::isa<mlir::LLVM::LLVMPointerType>(srcType)) {
      return clang_mlir::pointerToInteger(builder, loc, value, targetType);
    }

    if (mlir::isa<mlir::IntegerType>(srcType) ||
        mlir::isa<mlir::FloatType>(srcType)) {
      return clang_mlir::toValue(builder, loc, value, targetType,
                                 !isUnsignedIntegerLike(arg.clangType));
    }
  }

  if (mlir::isa<mlir::FloatType>(targetType)) {
    if (srcType.isIndex()) {
      value = mlir::arith::IndexCastOp::create(builder, loc,
                                               builder.getI64Type(), value)
                  .getResult();
    }

    if (mlir::isa<mlir::IntegerType>(value.getType()) ||
        mlir::isa<mlir::FloatType>(value.getType())) {
      return clang_mlir::toValue(builder, loc, value, targetType,
                                 !isUnsignedIntegerLike(arg.clangType));
    }
  }

  mlir::emitError(loc)
      << "clang-to-mlir: cannot coerce external C ABI argument: " << srcType
      << " -> " << targetType << "\n";
  return nullptr;
}

static auto promoteVariadicArg(mlir::OpBuilder &builder, mlir::Location loc,
                               const CallArg &arg) -> mlir::Value {
  mlir::Value value = arg.value;
  if (!value) {
    return {};
  }
  mlir::Type type = value.getType();

  if (mlir::isa<mlir::MemRefType>(type)) {
    return clang_mlir::memrefToPointer(
        builder, loc, value,
        mlir::LLVM::LLVMPointerType::get(builder.getContext()));
  }

  if (mlir::isa<mlir::LLVM::LLVMPointerType>(type)) {
    return value;
  }

  if (type.isIndex()) {
    return mlir::arith::IndexCastOp::create(builder, loc, builder.getI64Type(),
                                            value)
        .getResult();
  }

  if (auto intType = mlir::dyn_cast<mlir::IntegerType>(type)) {
    if (intType.getWidth() < 32) {
      if (isUnsignedIntegerLike(arg.clangType)) {
        return mlir::arith::ExtUIOp::create(builder, loc, builder.getI32Type(),
                                            value)
            .getResult();
      }

      return mlir::arith::ExtSIOp::create(builder, loc, builder.getI32Type(),
                                          value)
          .getResult();
    }

    return value;
  }

  if (auto floatType = mlir::dyn_cast<mlir::FloatType>(type)) {
    if (floatType.getWidth() < 64) {
      return mlir::arith::ExtFOp::create(builder, loc, builder.getF64Type(),
                                         value, {})
          .getResult();
    }

    return value;
  }

  if (isScalarLLVMCompatible(type)) {
    return value;
  }

  mlir::emitError(loc)
      << "clang-to-mlir: unsupported variadic C ABI argument type: " << type
      << "\n";
  return nullptr;
}

auto coerceFuncCallArg(mlir::OpBuilder &builder, mlir::Location loc,
                       const CallArg &arg, mlir::Type declaredType)
    -> mlir::Value {
  mlir::Value value = arg.value;
  if (!value) {
    return {};
  }
  mlir::Type actualType = value.getType();

  if (actualType == declaredType) {
    return value;
  }

  auto actualMemref = mlir::dyn_cast<mlir::MemRefType>(actualType);
  auto declaredMemref = mlir::dyn_cast<mlir::MemRefType>(declaredType);

  if (actualMemref && declaredMemref) {
    return clang_mlir::coerceMemRefForCall(builder, loc, value, declaredMemref);
  }

  if (mlir::isa<mlir::LLVM::LLVMPointerType>(declaredType)) {
    return materializeLlvmPointer(builder, loc, value);
  }

  if (actualType.isIndex() || mlir::isa<mlir::IntegerType>(actualType) ||
      mlir::isa<mlir::FloatType>(actualType) ||
      mlir::isa<mlir::LLVM::LLVMPointerType>(actualType)) {
    return coerceExternalArg(builder, loc, arg, declaredType);
  }

  mlir::emitError(loc) << "clang-to-mlir: cannot coerce function argument: "
                       << actualType << " -> " << declaredType << "\n";
  return nullptr;
}

auto emitExternalCall(mlir::OpBuilder &builder, mlir::Location loc,
                      mlir::ModuleOp module, llvm::StringRef name,
                      llvm::ArrayRef<CallArg> args,
                      llvm::ArrayRef<mlir::Type> fixedArgTypes,
                      mlir::Type retType, bool isVarArg, mlir::Value &outResult)
    -> bool {
  auto funcType =
      mlir::LLVM::LLVMFunctionType::get(retType, fixedArgTypes, isVarArg);

  if (!getOrCreateLLVMFunctionDecl(builder, module, name, funcType)) {
    return true;
  }

  if (!isVarArg && args.size() != fixedArgTypes.size()) {
    mlir::emitError(loc) << "clang-to-mlir: function '" << name << "' expects "
                         << fixedArgTypes.size() << " argument(s), got "
                         << args.size() << "\n";
    outResult = nullptr;
    return true;
  }

  if (isVarArg && args.size() < fixedArgTypes.size()) {
    mlir::emitError(loc) << "clang-to-mlir: variadic function '" << name
                         << "' expects at least " << fixedArgTypes.size()
                         << " fixed argument(s), got " << args.size() << "\n";
    outResult = nullptr;
    return true;
  }

  llvm::SmallVector<mlir::Value, 8> preparedArgs;
  preparedArgs.reserve(args.size());

  for (size_t i = 0; i < args.size(); ++i) {
    mlir::Value prepared;

    if (i < fixedArgTypes.size()) {
      prepared = coerceExternalArg(builder, loc, args[i], fixedArgTypes[i]);
    } else {
      prepared = promoteVariadicArg(builder, loc, args[i]);
    }

    if (!prepared) {
      outResult = nullptr;
      return true;
    }

    preparedArgs.push_back(prepared);
  }

  bool isVoid = mlir::isa<mlir::LLVM::LLVMVoidType>(retType);
  llvm::SmallVector<mlir::Type, 1> resultTypes;

  if (!isVoid) {
    resultTypes.push_back(retType);
  }

  auto callOp = mlir::LLVM::CallOp::create(
      builder, loc, resultTypes,
      mlir::FlatSymbolRefAttr::get(builder.getContext(), name), preparedArgs);

  if (isVarArg) {
    callOp->setAttr("var_callee_type", mlir::TypeAttr::get(funcType));
  }

  outResult =
      (!isVoid && callOp.getNumResults() > 0) ? callOp.getResult() : nullptr;
  return true;
}

} // namespace cmlir::frontend
