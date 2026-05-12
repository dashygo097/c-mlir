#include "../../Converter.h"
#include "../Utils/Casts.h"
#include "../Utils/Constants.h"
#include "../Utils/MemrefABI.h"
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

namespace cmlirc {

auto getOrCreateFunctionDecl(mlir::OpBuilder &builder, mlir::ModuleOp module,
                             const std::string &name,
                             mlir::FunctionType funcType)
    -> mlir::func::FuncOp {
  if (auto existing = module.lookupSymbol<mlir::func::FuncOp>(name)) {
    return existing;
  }

  mlir::OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPointToStart(module.getBody());

  auto funcOp = mlir::func::FuncOp::create(builder, builder.getUnknownLoc(),
                                           name, funcType);
  funcOp.setPrivate();

  return funcOp;
}

struct CallArg {
  mlir::Value value;
  clang::QualType clangType;
};

auto getOrCreateLLVMFunctionDecl(mlir::OpBuilder &builder,
                                 mlir::ModuleOp module, llvm::StringRef name,
                                 mlir::LLVM::LLVMFunctionType funcType)
    -> mlir::LLVM::LLVMFuncOp {
  if (auto existing = module.lookupSymbol<mlir::LLVM::LLVMFuncOp>(name)) {
    return existing;
  }

  mlir::OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPointToStart(module.getBody());

  return mlir::LLVM::LLVMFuncOp::create(builder, builder.getUnknownLoc(), name,
                                        funcType,
                                        mlir::LLVM::Linkage::External);
}

auto isUnsignedIntegerLike(clang::QualType type) -> bool {
  if (type.isNull()) {
    return false;
  }

  type = type.getCanonicalType();

  if (type->isBooleanType()) {
    return true;
  }

  return type->isUnsignedIntegerType();
}

auto isScalarLLVMCompatible(mlir::Type type) -> bool {
  return mlir::isa<mlir::IntegerType>(type) ||
         mlir::isa<mlir::FloatType>(type) ||
         mlir::isa<mlir::LLVM::LLVMPointerType>(type);
}

auto indexToInteger(mlir::OpBuilder &builder, mlir::Location loc,
                    mlir::Value value, mlir::IntegerType targetType)
    -> mlir::Value {
  return mlir::arith::IndexCastOp::create(builder, loc, targetType, value)
      .getResult();
}

auto integerToPointer(mlir::OpBuilder &builder, mlir::Location loc,
                      mlir::Value value, mlir::Type ptrType) -> mlir::Value {
  if (value.getType().isIndex()) {
    value = mlir::arith::IndexCastOp::create(builder, loc, builder.getI64Type(),
                                             value)
                .getResult();
  } else if (mlir::isa<mlir::IntegerType>(value.getType()) &&
             value.getType() != builder.getI64Type()) {
    value = utils::toValue(builder, loc, value, builder.getI64Type(), false);
  }

  return mlir::LLVM::IntToPtrOp::create(builder, loc, ptrType, value)
      .getResult();
}

auto pointerToInteger(mlir::OpBuilder &builder, mlir::Location loc,
                      mlir::Value value, mlir::Type intType) -> mlir::Value {
  return mlir::LLVM::PtrToIntOp::create(builder, loc, intType, value)
      .getResult();
}

auto memrefToLlvmPointer(mlir::OpBuilder &builder, mlir::Location loc,
                         mlir::Value value) -> mlir::Value {
  auto ptrType = mlir::LLVM::LLVMPointerType::get(builder.getContext());

  mlir::Value ptrAsIndex = mlir::memref::ExtractAlignedPointerAsIndexOp::create(
                               builder, loc, builder.getIndexType(), value)
                               .getResult();

  mlir::Value ptrAsI64 = mlir::arith::IndexCastOp::create(
                             builder, loc, builder.getI64Type(), ptrAsIndex)
                             .getResult();

  return mlir::LLVM::IntToPtrOp::create(builder, loc, ptrType, ptrAsI64)
      .getResult();
}

auto materializeLlvmPointer(mlir::OpBuilder &builder, mlir::Location loc,
                            mlir::Value value) -> mlir::Value {
  mlir::Type type = value.getType();
  auto ptrType = mlir::LLVM::LLVMPointerType::get(builder.getContext());

  if (mlir::isa<mlir::LLVM::LLVMPointerType>(type)) {
    return value;
  }

  if (mlir::isa<mlir::MemRefType>(type)) {
    return memrefToLlvmPointer(builder, loc, value);
  }

  if (type.isIndex() || mlir::isa<mlir::IntegerType>(type)) {
    return integerToPointer(builder, loc, value, ptrType);
  }

  llvm::WithColor::error()
      << "cmlirc: cannot materialize LLVM pointer from value of type " << type
      << "\n";
  return nullptr;
}

auto coerceExternalArg(mlir::OpBuilder &builder, mlir::Location loc,
                       const CallArg &arg, mlir::Type targetType)
    -> mlir::Value {
  mlir::Value value = arg.value;
  mlir::Type srcType = value.getType();

  if (srcType == targetType) {
    return value;
  }

  if (mlir::isa<mlir::LLVM::LLVMPointerType>(targetType)) {
    return materializeLlvmPointer(builder, loc, value);
  }

  if (mlir::isa<mlir::MemRefType>(srcType)) {
    llvm::WithColor::error()
        << "cmlirc: cannot pass memref value to non-pointer C ABI argument: "
        << srcType << " -> " << targetType << "\n";
    return nullptr;
  }

  if (auto targetInt = mlir::dyn_cast<mlir::IntegerType>(targetType)) {
    if (srcType.isIndex()) {
      return indexToInteger(builder, loc, value, targetInt);
    }

    if (mlir::isa<mlir::LLVM::LLVMPointerType>(srcType)) {
      return pointerToInteger(builder, loc, value, targetType);
    }

    if (mlir::isa<mlir::IntegerType>(srcType) ||
        mlir::isa<mlir::FloatType>(srcType)) {
      return utils::toValue(builder, loc, value, targetType,
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
      return utils::toValue(builder, loc, value, targetType,
                              !isUnsignedIntegerLike(arg.clangType));
    }
  }

  llvm::WithColor::error() << "cmlirc: cannot coerce external C ABI argument: "
                           << srcType << " -> " << targetType << "\n";
  return nullptr;
}

auto promoteVariadicArg(mlir::OpBuilder &builder, mlir::Location loc,
                        const CallArg &arg) -> mlir::Value {
  mlir::Value value = arg.value;
  mlir::Type type = value.getType();

  if (mlir::isa<mlir::MemRefType>(type)) {
    return memrefToLlvmPointer(builder, loc, value);
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
                                         value)
          .getResult();
    }

    return value;
  }

  if (isScalarLLVMCompatible(type)) {
    return value;
  }

  llvm::WithColor::error()
      << "cmlirc: unsupported variadic C ABI argument type: " << type << "\n";
  return nullptr;
}

auto coerceFuncCallArg(mlir::OpBuilder &builder, mlir::Location loc,
                       const CallArg &arg, mlir::Type declaredType)
    -> mlir::Value {
  mlir::Value value = arg.value;
  mlir::Type actualType = value.getType();

  if (actualType == declaredType) {
    return value;
  }

  auto actualMemref = mlir::dyn_cast<mlir::MemRefType>(actualType);
  auto declaredMemref = mlir::dyn_cast<mlir::MemRefType>(declaredType);

  if (actualMemref && declaredMemref) {
    return utils::coerceMemRefForCall(builder, loc, value, declaredMemref);
  }

  if (mlir::isa<mlir::LLVM::LLVMPointerType>(declaredType)) {
    return materializeLlvmPointer(builder, loc, value);
  }

  if (actualType.isIndex() || mlir::isa<mlir::IntegerType>(actualType) ||
      mlir::isa<mlir::FloatType>(actualType) ||
      mlir::isa<mlir::LLVM::LLVMPointerType>(actualType)) {
    return coerceExternalArg(builder, loc, arg, declaredType);
  }

  llvm::WithColor::error() << "cmlirc: cannot coerce function argument: "
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

  getOrCreateLLVMFunctionDecl(builder, module, name, funcType);

  if (!isVarArg && args.size() != fixedArgTypes.size()) {
    llvm::WithColor::error()
        << "cmlirc: function '" << name << "' expects " << fixedArgTypes.size()
        << " argument(s), got " << args.size() << "\n";
    outResult = nullptr;
    return true;
  }

  if (isVarArg && args.size() < fixedArgTypes.size()) {
    llvm::WithColor::error()
        << "cmlirc: variadic function '" << name << "' expects at least "
        << fixedArgTypes.size() << " fixed argument(s), got " << args.size()
        << "\n";
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

auto tryEmitStdlibCall(mlir::OpBuilder &builder, mlir::Location loc,
                       mlir::ModuleOp module, llvm::StringRef name,
                       llvm::ArrayRef<CallArg> args, mlir::Value &outResult)
    -> bool {
  mlir::MLIRContext *ctx = builder.getContext();

  auto i32 = builder.getI32Type();
  auto i64 = builder.getI64Type();
  auto f64 = builder.getF64Type();
  auto addr = mlir::LLVM::LLVMPointerType::get(ctx);
  auto voidTy = mlir::LLVM::LLVMVoidType::get(ctx);

  auto emit = [&](llvm::ArrayRef<mlir::Type> fixedArgTypes, mlir::Type retType,
                  bool isVarArg) -> bool {
    return emitExternalCall(builder, loc, module, name, args, fixedArgTypes,
                            retType, isVarArg, outResult);
  };

#define VOID0() return emit(llvm::ArrayRef<mlir::Type>{}, voidTy, false)
#define RET0(r) return emit(llvm::ArrayRef<mlir::Type>{}, r, false)
#define VOID(...) return emit({__VA_ARGS__}, voidTy, false)
#define RET(r, ...) return emit({__VA_ARGS__}, r, false)
#define VRET(r, ...) return emit({__VA_ARGS__}, r, true)

  if (name == "printf")
    VRET(i32, addr);
  if (name == "fprintf")
    VRET(i32, addr, addr);
  if (name == "sprintf")
    VRET(i32, addr, addr);
  if (name == "snprintf")
    VRET(i32, addr, i64, addr);
  if (name == "scanf")
    VRET(i32, addr);
  if (name == "fscanf")
    VRET(i32, addr, addr);
  if (name == "sscanf")
    VRET(i32, addr, addr);
  if (name == "puts")
    RET(i32, addr);
  if (name == "putchar")
    RET(i32, i32);
  if (name == "getchar")
    RET0(i32);
  if (name == "gets")
    RET(addr, addr);
  if (name == "fgets")
    RET(addr, addr, i32, addr);
  if (name == "fputs")
    RET(i32, addr, addr);
  if (name == "fopen")
    RET(addr, addr, addr);
  if (name == "fclose")
    RET(i32, addr);
  if (name == "fread")
    RET(i64, addr, i64, i64, addr);
  if (name == "fwrite")
    RET(i64, addr, i64, i64, addr);
  if (name == "fseek")
    RET(i32, addr, i64, i32);
  if (name == "ftell")
    RET(i64, addr);
  if (name == "rewind")
    VOID(addr);
  if (name == "feof")
    RET(i32, addr);
  if (name == "ferror")
    RET(i32, addr);
  if (name == "fflush")
    RET(i32, addr);
  if (name == "remove")
    RET(i32, addr);
  if (name == "rename")
    RET(i32, addr, addr);
  if (name == "perror")
    VOID(addr);

  if (name == "malloc")
    RET(addr, i64);
  if (name == "calloc")
    RET(addr, i64, i64);
  if (name == "realloc")
    RET(addr, addr, i64);
  if (name == "free")
    VOID(addr);
  if (name == "exit")
    VOID(i32);
  if (name == "abort")
    VOID0();
  if (name == "rand")
    RET0(i32);
  if (name == "srand")
    VOID(i32);
  if (name == "atoi")
    RET(i32, addr);
  if (name == "atol")
    RET(i64, addr);
  if (name == "atof")
    RET(f64, addr);
  if (name == "strtol")
    RET(i64, addr, addr, i32);
  if (name == "strtod")
    RET(f64, addr, addr);
  if (name == "qsort")
    VOID(addr, i64, i64, addr);
  if (name == "bsearch")
    RET(addr, addr, addr, i64, i64, addr);
  if (name == "getenv")
    RET(addr, addr);
  if (name == "system")
    RET(i32, addr);

  if (name == "strlen")
    RET(i64, addr);
  if (name == "strcpy")
    RET(addr, addr, addr);
  if (name == "strncpy")
    RET(addr, addr, addr, i64);
  if (name == "strcat")
    RET(addr, addr, addr);
  if (name == "strncat")
    RET(addr, addr, addr, i64);
  if (name == "strcmp")
    RET(i32, addr, addr);
  if (name == "strncmp")
    RET(i32, addr, addr, i64);
  if (name == "strchr")
    RET(addr, addr, i32);
  if (name == "strrchr")
    RET(addr, addr, i32);
  if (name == "strstr")
    RET(addr, addr, addr);
  if (name == "strtok")
    RET(addr, addr, addr);
  if (name == "strdup")
    RET(addr, addr);
  if (name == "memcpy")
    RET(addr, addr, addr, i64);
  if (name == "memmove")
    RET(addr, addr, addr, i64);
  if (name == "memset")
    RET(addr, addr, i32, i64);
  if (name == "memcmp")
    RET(i32, addr, addr, i64);
  if (name == "memchr")
    RET(addr, addr, i32, i64);

  if (name == "modf" || name == "modff")
    RET(f64, f64, addr);
  if (name == "frexp" || name == "frexpf")
    RET(f64, f64, addr);
  if (name == "ldexp" || name == "ldexpf")
    RET(f64, f64, i32);
  if (name == "hypot" || name == "hypotf")
    RET(f64, f64, f64);
  if (name == "fmod" || name == "fmodf")
    RET(f64, f64, f64);
  if (name == "remainder" || name == "remainderf")
    RET(f64, f64, f64);
  if (name == "fmin" || name == "fminf")
    RET(f64, f64, f64);
  if (name == "fmax" || name == "fmaxf")
    RET(f64, f64, f64);
  if (name == "fdim" || name == "fdimf")
    RET(f64, f64, f64);
  if (name == "nearbyint" || name == "nearbyintf")
    RET(f64, f64);
  if (name == "rint" || name == "rintf")
    RET(f64, f64);
  if (name == "lround" || name == "lroundf")
    RET(i64, f64);
  if (name == "lrint" || name == "lrintf")
    RET(i64, f64);
  if (name == "scalbn" || name == "scalbnf")
    RET(f64, f64, i32);
  if (name == "ilogb" || name == "ilogbf")
    RET(i32, f64);
  if (name == "logb" || name == "logbf")
    RET(f64, f64);
  if (name == "nan" || name == "nanf")
    RET(f64, addr);

  if (name == "time")
    RET(i64, addr);
  if (name == "clock")
    RET0(i64);
  if (name == "difftime")
    RET(f64, i64, i64);
  if (name == "mktime")
    RET(i64, addr);
  if (name == "gmtime")
    RET(addr, addr);
  if (name == "localtime")
    RET(addr, addr);
  if (name == "strftime")
    RET(i64, addr, i64, addr, addr);
  if (name == "asctime")
    RET(addr, addr);
  if (name == "ctime")
    RET(addr, addr);

  if (name == "read")
    RET(i64, i32, addr, i64);
  if (name == "write")
    RET(i64, i32, addr, i64);
  if (name == "close")
    RET(i32, i32);
  if (name == "open")
    VRET(i32, addr, i32);
  if (name == "sleep")
    RET(i32, i32);
  if (name == "usleep")
    RET(i32, i32);
  if (name == "getpid")
    RET0(i32);
  if (name == "getppid")
    RET0(i32);
  if (name == "fork")
    RET0(i32);
  if (name == "execv")
    RET(i32, addr, addr);
  if (name == "execvp")
    RET(i32, addr, addr);

#undef VOID0
#undef RET0
#undef VOID
#undef RET
#undef VRET

  return false;
}

auto matchCall(llvm::StringRef callee, llvm::StringRef pattern) -> bool {
  llvm::SmallVector<llvm::StringRef, 4> tokens;
  pattern.split(tokens, '|');
  return llvm::is_contained(tokens, callee);
}

auto alignArithmeticArgs(mlir::OpBuilder &builder, mlir::Location loc,
                         std::vector<mlir::Value> &args, bool forceFloat)
    -> bool {
  if (args.empty()) {
    return true;
  }

  mlir::Type targetType = args[0].getType();

  if (forceFloat) {
    bool hasFloat = false;

    for (auto value : args) {
      if (!mlir::isa<mlir::IntegerType>(value.getType()) &&
          !mlir::isa<mlir::FloatType>(value.getType())) {
        return false;
      }

      if (mlir::isa<mlir::FloatType>(value.getType())) {
        if (!hasFloat || value.getType().getIntOrFloatBitWidth() >
                             targetType.getIntOrFloatBitWidth()) {
          targetType = value.getType();
        }
        hasFloat = true;
      }
    }

    if (!hasFloat) {
      targetType = builder.getF64Type();
    }
  } else {
    for (auto value : args) {
      if (!mlir::isa<mlir::IntegerType>(value.getType()) &&
          !mlir::isa<mlir::FloatType>(value.getType())) {
        return false;
      }

      if (value.getType().getIntOrFloatBitWidth() >
          targetType.getIntOrFloatBitWidth()) {
        targetType = value.getType();
      }
    }
  }

  for (auto &value : args) {
    value = utils::toValue(builder, loc, value, targetType, true);
  }

  return true;
}

mlir::Value CMLIRConverter::generateCallExpr(clang::CallExpr *callExpr) {
  mlir::ModuleOp module = contextManager.Module();
  mlir::OpBuilder &builder = contextManager.Builder();
  mlir::Location loc = builder.getUnknownLoc();

  const clang::FunctionDecl *calleeDecl = callExpr->getDirectCallee();
  if (!calleeDecl) {
    llvm::WithColor::error() << "cmlirc: indirect calls not supported\n";
    return nullptr;
  }

  std::string calleeName = calleeDecl->getNameAsString();
  uint32_t numArgs = callExpr->getNumArgs();

#define REGISTER_FLOAT_BYPASS(OpClass, names)                                  \
  if (matchCall(calleeName, names)) {                                          \
    std::vector<mlir::Value> args;                                             \
    for (uint32_t i = 0; i < numArgs; ++i) {                                   \
      mlir::Value arg = generateExpr(callExpr->getArg(i));                     \
      args.push_back(arg);                                                     \
    }                                                                          \
    if (!alignArithmeticArgs(builder, loc, args, true)) {                      \
      llvm::WithColor::error()                                                 \
          << "cmlirc: invalid argument type for math call '" << calleeName     \
          << "'\n";                                                            \
      return nullptr;                                                          \
    }                                                                          \
    return OpClass::create(builder, loc, args).getResult();                    \
  }

#define REGISTER_INT_BYPASS(OpClass, names)                                    \
  if (matchCall(calleeName, names)) {                                          \
    std::vector<mlir::Value> args;                                             \
    for (uint32_t i = 0; i < numArgs; ++i) {                                   \
      mlir::Value arg = generateExpr(callExpr->getArg(i));                     \
      args.push_back(arg);                                                     \
    }                                                                          \
    if (!alignArithmeticArgs(builder, loc, args, false)) {                     \
      llvm::WithColor::error()                                                 \
          << "cmlirc: invalid argument type for math call '" << calleeName     \
          << "'\n";                                                            \
      return nullptr;                                                          \
    }                                                                          \
    return OpClass::create(builder, loc, args).getResult();                    \
  }

#define REGISTER_OVERLOAD_BYPASS(IntOpClass, FloatOpClass, names)              \
  if (matchCall(calleeName, names)) {                                          \
    std::vector<mlir::Value> args;                                             \
    bool hasFloat = false;                                                     \
    for (uint32_t i = 0; i < numArgs; ++i) {                                   \
      mlir::Value arg = generateExpr(callExpr->getArg(i));                     \
      if (mlir::isa<mlir::FloatType>(arg.getType())) {                         \
        hasFloat = true;                                                       \
      }                                                                        \
      args.push_back(arg);                                                     \
    }                                                                          \
    if (!alignArithmeticArgs(builder, loc, args, hasFloat)) {                  \
      llvm::WithColor::error()                                                 \
          << "cmlirc: invalid argument type for math call '" << calleeName     \
          << "'\n";                                                            \
      return nullptr;                                                          \
    }                                                                          \
    if (hasFloat) {                                                            \
      return FloatOpClass::create(builder, loc, args).getResult();             \
    }                                                                          \
    return IntOpClass::create(builder, loc, args).getResult();                 \
  }

  REGISTER_OVERLOAD_BYPASS(mlir::arith::MinSIOp, mlir::arith::MinNumFOp,
                           "min|mini|minl|fmin|fminf|fminl")
  REGISTER_OVERLOAD_BYPASS(mlir::arith::MaxSIOp, mlir::arith::MaxNumFOp,
                           "max|maxi|maxl|fmax|fmaxf|fmaxl")
  REGISTER_OVERLOAD_BYPASS(mlir::math::AbsIOp, mlir::math::AbsFOp,
                           "abs|absi|absl|fabs|fabsf|fabsl")
  REGISTER_FLOAT_BYPASS(mlir::math::AcosOp, "acos|acosf|acosl")
  REGISTER_FLOAT_BYPASS(mlir::math::AcoshOp, "acosh|acoshf|acoshl")
  REGISTER_FLOAT_BYPASS(mlir::math::AsinOp, "asin|asinf|asinl")
  REGISTER_FLOAT_BYPASS(mlir::math::AsinhOp, "asinh|asinhf|asinhl")
  REGISTER_FLOAT_BYPASS(mlir::math::AtanOp, "atan|atanf|atanl")
  REGISTER_FLOAT_BYPASS(mlir::math::Atan2Op, "atan2|atan2f|atan2l")
  REGISTER_FLOAT_BYPASS(mlir::math::AtanhOp, "atanh|atanhf|atanhl")
  REGISTER_FLOAT_BYPASS(mlir::math::CbrtOp, "cbrt|cbrtf|cbrtl")
  REGISTER_FLOAT_BYPASS(mlir::math::CeilOp, "ceil|ceilf|ceill")
  REGISTER_FLOAT_BYPASS(mlir::math::CopySignOp, "copysign|copysignf|copysignl")
  REGISTER_FLOAT_BYPASS(mlir::math::CosOp, "cos|cosf|cosl")
  REGISTER_FLOAT_BYPASS(mlir::math::CoshOp, "cosh|coshf|coshl")
  REGISTER_INT_BYPASS(mlir::math::CountLeadingZerosOp, "ctlz|ctlzl")
  REGISTER_INT_BYPASS(mlir::math::CtPopOp, "ctpop|ctpopl")
  REGISTER_INT_BYPASS(mlir::math::CountTrailingZerosOp, "cttz|cttzl")
  REGISTER_FLOAT_BYPASS(mlir::math::ErfOp, "erf|erff|erfl")
  REGISTER_FLOAT_BYPASS(mlir::math::ErfcOp, "erfc|erfcf|erfcl")
  REGISTER_FLOAT_BYPASS(mlir::math::ExpOp, "exp|expf|expl")
  REGISTER_FLOAT_BYPASS(mlir::math::Exp2Op, "exp2|exp2f|exp2l")
  REGISTER_FLOAT_BYPASS(mlir::math::ExpM1Op, "expm1|expm1f|expm1l")
  REGISTER_FLOAT_BYPASS(mlir::math::FloorOp, "floor|floorf|floorl")
  REGISTER_FLOAT_BYPASS(mlir::math::FmaOp, "fma|fmaf|fmal")
  REGISTER_FLOAT_BYPASS(mlir::math::IsFiniteOp, "isfinite|isfinitef|isfinitel")
  REGISTER_FLOAT_BYPASS(mlir::math::IsInfOp, "isinf|isinff|isinfl")
  REGISTER_FLOAT_BYPASS(mlir::math::IsNaNOp, "isnan|isnanf|isnanl")
  REGISTER_FLOAT_BYPASS(mlir::math::IsNormalOp, "isnormal|isnormalf|isnormall")
  REGISTER_FLOAT_BYPASS(mlir::math::LogOp, "log|logf|logl")
  REGISTER_FLOAT_BYPASS(mlir::math::Log10Op, "log10|log10f|log10l")
  REGISTER_FLOAT_BYPASS(mlir::math::Log1pOp, "log1p|log1pf|log1pl")
  REGISTER_FLOAT_BYPASS(mlir::math::Log2Op, "log2|log2f|log2l")
  REGISTER_FLOAT_BYPASS(mlir::math::PowFOp, "pow|powf|powl")
  REGISTER_FLOAT_BYPASS(mlir::math::RoundOp, "round|roundf|roundl")
  REGISTER_FLOAT_BYPASS(mlir::math::RoundEvenOp,
                        "roundeven|roundevenf|roundevenl")
  REGISTER_FLOAT_BYPASS(mlir::math::RsqrtOp, "rsqrt|rsqrtf|rsqrtl")
  REGISTER_FLOAT_BYPASS(mlir::math::SinOp, "sin|sinf|sinl")
  REGISTER_FLOAT_BYPASS(mlir::math::SinhOp, "sinh|sinhf|sinhl")
  REGISTER_FLOAT_BYPASS(mlir::math::SqrtOp, "sqrt|sqrtf|sqrtl")
  REGISTER_FLOAT_BYPASS(mlir::math::TanOp, "tan|tanf|tanl")
  REGISTER_FLOAT_BYPASS(mlir::math::TanhOp, "tanh|tanhf|tanhl")
  REGISTER_FLOAT_BYPASS(mlir::math::TruncOp, "trunc|truncf|truncl")

#undef REGISTER_FLOAT_BYPASS
#undef REGISTER_INT_BYPASS
#undef REGISTER_OVERLOAD_BYPASS

  llvm::SmallVector<CallArg, 8> argValues;

  if (auto *memberCall = llvm::dyn_cast<clang::CXXMemberCallExpr>(callExpr)) {
    if (auto *methodDecl = llvm::dyn_cast<clang::CXXMethodDecl>(calleeDecl)) {
      if (!methodDecl->isStatic()) {
        clang::Expr *implicitObj = memberCall->getImplicitObjectArgument();
        mlir::Value thisVal = generateExpr(implicitObj);

        if (!mlir::isa<mlir::LLVM::LLVMPointerType>(thisVal.getType())) {
          mlir::Type structType = thisVal.getType();
          mlir::Value one =
              utils::intConst(builder, loc, builder.getI32Type(), 1);
          auto ptrType = mlir::LLVM::LLVMPointerType::get(builder.getContext());

          mlir::Value tempPtr = mlir::LLVM::AllocaOp::create(
              builder, loc, ptrType, structType, one);

          mlir::LLVM::StoreOp::create(builder, loc, thisVal, tempPtr);
          thisVal = tempPtr;
        }

        argValues.push_back({thisVal, implicitObj->getType()});
      }
    }
  }

  for (uint32_t i = 0; i < numArgs; ++i) {
    clang::Expr *argExpr = callExpr->getArg(i);
    mlir::Value value = generateExpr(argExpr);

    argValues.push_back({value, argExpr->getType()});
  }

  mlir::Value stdlibResult;
  if (tryEmitStdlibCall(builder, loc, module, calleeName, argValues,
                        stdlibResult)) {
    return stdlibResult;
  }

  llvm::SmallVector<mlir::Type, 8> declaredArgTypes;

  bool hasImplicitThis = false;
  if (auto *methodDecl = llvm::dyn_cast<clang::CXXMethodDecl>(calleeDecl)) {
    hasImplicitThis = !methodDecl->isStatic();
  }

  if (hasImplicitThis && !argValues.empty()) {
    declaredArgTypes.push_back(argValues.front().value.getType());
  }

  if (calleeDecl->param_size() == numArgs) {
    for (const clang::ParmVarDecl *param : calleeDecl->parameters()) {
      declaredArgTypes.push_back(convertType(param->getType()));
    }
  } else {
    for (const CallArg &arg : argValues) {
      declaredArgTypes.push_back(arg.value.getType());
    }
  }

  clang::QualType returnType = calleeDecl->getReturnType();
  mlir::Type mlirReturnType = convertType(returnType);

  llvm::SmallVector<mlir::Type, 1> returnTypes;
  if (!mlir::isa<mlir::NoneType>(mlirReturnType)) {
    returnTypes.push_back(mlirReturnType);
  }

  auto funcType = builder.getFunctionType(declaredArgTypes, returnTypes);
  auto funcDecl =
      getOrCreateFunctionDecl(builder, module, calleeName, funcType);

  llvm::SmallVector<mlir::Value, 8> castArgs;
  auto declaredFuncType = funcDecl.getFunctionType();

  if (declaredFuncType.getNumInputs() != argValues.size()) {
    llvm::WithColor::error()
        << "cmlirc: function '" << calleeName << "' expects "
        << declaredFuncType.getNumInputs() << " argument(s), got "
        << argValues.size() << "\n";
    return nullptr;
  }

  for (size_t i = 0; i < argValues.size(); ++i) {
    mlir::Value castArg = coerceFuncCallArg(builder, loc, argValues[i],
                                            declaredFuncType.getInput(i));

    if (!castArg) {
      return nullptr;
    }

    castArgs.push_back(castArg);
  }

  auto callOp = mlir::func::CallOp::create(builder, loc, calleeName,
                                           returnTypes, castArgs);
  return callOp.getNumResults() > 0 ? callOp.getResult(0) : nullptr;
}

} // namespace cmlirc
