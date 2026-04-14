#include "../../Converter.h"
#include "../Utils/Constants.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
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

auto tryEmitStdlibCall(mlir::OpBuilder &builder, mlir::Location loc,
                       mlir::ModuleOp module, const std::string &name,
                       llvm::ArrayRef<mlir::Value> callArgs,
                       mlir::Value &outResult) -> bool {
  mlir::MLIRContext *ctx = builder.getContext();

  auto i32 = mlir::IntegerType::get(ctx, 32);
  auto i64 = mlir::IntegerType::get(ctx, 64);
  // auto f32 = mlir::Float32Type::get(ctx);
  auto f64 = mlir::Float64Type::get(ctx);
  auto addr = mlir::LLVM::LLVMPointerType::get(ctx);

  // emit: fixed-arity functions — func.func private + func.call.
  auto emit = [&](llvm::ArrayRef<mlir::Type> declArgTypes,
                  llvm::ArrayRef<mlir::Type> retTypes) -> bool {
    auto funcType = builder.getFunctionType(declArgTypes, retTypes);
    getOrCreateFunctionDecl(builder, module, name, funcType);

    llvm::SmallVector<mlir::Value> promotedArgs;
    for (auto [arg, declTy] : llvm::zip(callArgs, declArgTypes)) {
      mlir::Type argTy = arg.getType();
      if (argTy == declTy) {
        promotedArgs.push_back(arg);
        continue;
      }
      // int → float
      if (mlir::isa<mlir::IntegerType>(argTy) &&
          mlir::isa<mlir::FloatType>(declTy)) {
        promotedArgs.push_back(
            mlir::arith::SIToFPOp::create(builder, loc, declTy, arg)
                .getResult());
        continue;
      }
      // float → float
      if (mlir::isa<mlir::FloatType>(argTy) &&
          mlir::isa<mlir::FloatType>(declTy)) {
        auto srcW = mlir::cast<mlir::FloatType>(argTy).getWidth();
        auto dstW = mlir::cast<mlir::FloatType>(declTy).getWidth();
        if (srcW < dstW)
          promotedArgs.push_back(
              mlir::arith::ExtFOp::create(builder, loc, declTy, arg)
                  .getResult());
        else
          promotedArgs.push_back(
              mlir::arith::TruncFOp::create(builder, loc, declTy, arg)
                  .getResult());
        continue;
      }
      // int → int
      if (mlir::isa<mlir::IntegerType>(argTy) &&
          mlir::isa<mlir::IntegerType>(declTy)) {
        auto srcW = mlir::cast<mlir::IntegerType>(argTy).getWidth();
        auto dstW = mlir::cast<mlir::IntegerType>(declTy).getWidth();
        if (srcW < dstW) {
          promotedArgs.push_back(
              mlir::arith::ExtSIOp::create(builder, loc, declTy, arg)
                  .getResult());
        } else {
          promotedArgs.push_back(
              mlir::arith::TruncIOp::create(builder, loc, declTy, arg)
                  .getResult());
        }
        continue;
      }
      promotedArgs.push_back(arg); // fallback
    }

    llvm::SmallVector<mlir::Type, 1> retTys(retTypes.begin(), retTypes.end());
    auto callOp =
        mlir::func::CallOp::create(builder, loc, name, retTys, promotedArgs);
    outResult = callOp.getNumResults() > 0 ? callOp.getResult(0) : nullptr;
    return true;
  };

  // variadic functions (printf, scanf, ...).
  auto emitVariadic = [&](llvm::ArrayRef<mlir::Type> fixedArgTypes,
                          mlir::Type retType) -> bool {
    // Declare llvm.func @name(fixedArgs..., ...) -> retType  (idempotent)
    if (!module.lookupSymbol<mlir::LLVM::LLVMFuncOp>(name)) {
      mlir::OpBuilder::InsertionGuard guard(builder);
      builder.setInsertionPointToStart(module.getBody());
      auto llvmFuncTy =
          mlir::LLVM::LLVMFunctionType::get(retType, fixedArgTypes,
                                            /*isVarArg=*/true);
      mlir::LLVM::LLVMFuncOp::create(builder, builder.getUnknownLoc(), name,
                                     llvmFuncTy, mlir::LLVM::Linkage::External);
    }
    // Emit llvm.call ... vararg(...)
    auto llvmFuncOp = module.lookupSymbol<mlir::LLVM::LLVMFuncOp>(name);
    auto llvmFuncTy = llvmFuncOp.getFunctionType();
    bool isVoid = mlir::isa<mlir::LLVM::LLVMVoidType>(retType);
    mlir::SmallVector<mlir::Type, 1> resultTypes;
    if (!isVoid) {
      resultTypes.push_back(retType);
    }
    auto callOp = mlir::LLVM::CallOp::create(
        builder, loc, resultTypes,
        mlir::FlatSymbolRefAttr::get(builder.getContext(), name), callArgs);
    callOp->setAttr("var_callee_type", mlir::TypeAttr::get(llvmFuncTy));
    outResult =
        (!isVoid && callOp.getNumResults() > 0) ? callOp.getResult() : nullptr;
    return true;
  };

#define VOID(...) return emit({__VA_ARGS__}, {})
#define RET(r, ...) return emit({__VA_ARGS__}, {r})
#define VRET(r, ...) return emitVariadic({__VA_ARGS__}, r)
#define VVOID(...)                                                             \
  return emitVariadic({__VA_ARGS__}, mlir::LLVM::LLVMVoidType::get(ctx))

  // stdio.h
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
    RET(i32);
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

  // stdlib.h
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
    VOID();
  if (name == "rand")
    RET(i32);
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

  // string.h
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

  // math.h
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

  // time.h
  if (name == "time")
    RET(i64, addr);
  if (name == "clock")
    RET(i64);
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

  // unistd.h / POSIX
  if (name == "read")
    RET(i64, i32, addr, i64);
  if (name == "write")
    RET(i64, i32, addr, i64);
  if (name == "close")
    RET(i32, i32);
  if (name == "open")
    RET(i32, addr, i32);
  if (name == "sleep")
    RET(i32, i32);
  if (name == "usleep")
    RET(i32, i32);
  if (name == "getpid")
    RET(i32);
  if (name == "getppid")
    RET(i32);
  if (name == "fork")
    RET(i32);
  if (name == "execv")
    RET(i32, addr, addr);
  if (name == "execvp")
    RET(i32, addr, addr);

#undef VOID
#undef RET
#undef VRET
#undef VVOID

  return false;
}

// main entry point
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
  uint32_t num_args = callExpr->getNumArgs();
  std::vector<std::string> tokens;
  std::string token;

  // MLIR dialect ops

  auto matchCall = [&](llvm::StringRef callee, llvm::StringRef pattern) {
    llvm::SmallVector<llvm::StringRef, 4> tokens;
    pattern.split(tokens, '|');
    return llvm::is_contained(tokens, callee);
  };

  auto alignTypes = [&](std::vector<mlir::Value> &args, bool forceFloat) {
    if (args.empty())
      return;
    mlir::Type targetType = args[0].getType();

    if (forceFloat) {
      bool hasFloat = false;
      for (auto v : args) {
        if (mlir::isa<mlir::FloatType>(v.getType())) {
          if (!hasFloat || v.getType().getIntOrFloatBitWidth() >
                               targetType.getIntOrFloatBitWidth())
            targetType = v.getType();
          hasFloat = true;
        }
      }
      if (!hasFloat)
        targetType = builder.getF64Type();

      for (auto &v : args) {
        mlir::Type t = v.getType();
        if (mlir::isa<mlir::IntegerType>(t)) {
          v = mlir::arith::SIToFPOp::create(builder, loc, targetType, v);
        } else if (t != targetType) {
          if (t.getIntOrFloatBitWidth() < targetType.getIntOrFloatBitWidth())
            v = mlir::arith::ExtFOp::create(builder, loc, targetType, v);
          else
            v = mlir::arith::TruncFOp::create(builder, loc, targetType, v);
        }
      }
    } else {
      for (auto v : args) {
        if (v.getType().getIntOrFloatBitWidth() >
            targetType.getIntOrFloatBitWidth())
          targetType = v.getType();
      }
      for (auto &v : args) {
        mlir::Type t = v.getType();
        if (t != targetType) {
          if (t.getIntOrFloatBitWidth() < targetType.getIntOrFloatBitWidth())
            v = mlir::arith::ExtSIOp::create(builder, loc, targetType, v);
          else
            v = mlir::arith::TruncIOp::create(builder, loc, targetType, v);
        }
      }
    }
  };

#define REGISTER_FLOAT_BYPASS(OpClass, names)                                  \
  if (matchCall(calleeName, names)) {                                          \
    std::vector<mlir::Value> args;                                             \
    for (uint32_t i = 0; i < num_args; ++i) {                                  \
      mlir::Value arg = generateExpr(callExpr->getArg(i));                     \
      if (!arg) {                                                              \
        llvm::WithColor::error()                                               \
            << "cmlirc: failed to generate argument " << i << "\n";            \
        return nullptr;                                                        \
      }                                                                        \
      args.push_back(arg);                                                     \
    }                                                                          \
    alignTypes(args, /*forceFloat=*/true);                                     \
    return OpClass::create(builder, loc, args).getResult();                    \
  }

// Macro for Integer operations
#define REGISTER_INT_BYPASS(OpClass, names)                                    \
  if (matchCall(calleeName, names)) {                                          \
    std::vector<mlir::Value> args;                                             \
    for (uint32_t i = 0; i < num_args; ++i) {                                  \
      mlir::Value arg = generateExpr(callExpr->getArg(i));                     \
      if (!arg) {                                                              \
        llvm::WithColor::error()                                               \
            << "cmlirc: failed to generate argument " << i << "\n";            \
        return nullptr;                                                        \
      }                                                                        \
      args.push_back(arg);                                                     \
    }                                                                          \
    alignTypes(args, /*forceFloat=*/false);                                    \
    return OpClass::create(builder, loc, args).getResult();                    \
  }

#define REGISTER_OVERLOAD_BYPASS(IntOpClass, FloatOpClass, names)              \
  if (matchCall(calleeName, names)) {                                          \
    std::vector<mlir::Value> args;                                             \
    bool hasFloat = false;                                                     \
    for (uint32_t i = 0; i < num_args; ++i) {                                  \
      mlir::Value arg = generateExpr(callExpr->getArg(i));                     \
      if (!arg) {                                                              \
        llvm::WithColor::error()                                               \
            << "cmlirc: failed to generate argument " << i << "\n";            \
        return nullptr;                                                        \
      }                                                                        \
      if (mlir::isa<mlir::FloatType>(arg.getType()))                           \
        hasFloat = true;                                                       \
      args.push_back(arg);                                                     \
    }                                                                          \
    if (hasFloat) {                                                            \
      alignTypes(args, /*forceFloat=*/true);                                   \
      return FloatOpClass::create(builder, loc, args).getResult();             \
    } else {                                                                   \
      alignTypes(args, /*forceFloat=*/false);                                  \
      return IntOpClass::create(builder, loc, args).getResult();               \
    }                                                                          \
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

  // Build argument values
  llvm::SmallVector<mlir::Value, 4> argValues;

  // Check CXXMemberCallExpr
  if (auto *memberCall = mlir::dyn_cast<clang::CXXMemberCallExpr>(callExpr)) {
    if (auto *methodDecl = mlir::dyn_cast<clang::CXXMethodDecl>(calleeDecl)) {
      if (!methodDecl->isStatic()) {
        clang::Expr *implicitObj = memberCall->getImplicitObjectArgument();
        mlir::Value thisVal = generateExpr(implicitObj);
        if (!thisVal) {
          llvm::WithColor::error()
              << "cmlirc: failed to generate 'this' argument\n";
          return nullptr;
        }

        if (!mlir::isa<mlir::LLVM::LLVMPointerType>(thisVal.getType())) {
          mlir::Type structType = thisVal.getType();
          mlir::Value one =
              detail::intConst(builder, loc, builder.getI32Type(), 1);
          auto ptrType = mlir::LLVM::LLVMPointerType::get(builder.getContext());

          mlir::Value tempPtr = mlir::LLVM::AllocaOp::create(
              builder, loc, ptrType, structType, one);

          mlir::LLVM::StoreOp::create(builder, loc, thisVal, tempPtr);

          thisVal = tempPtr;
        }

        argValues.push_back(thisVal);
      }
    }
  }

  for (uint32_t i = 0; i < num_args; ++i) {
    mlir::Value v = generateExpr(callExpr->getArg(i));
    if (!v) {
      llvm::WithColor::error()
          << "cmlirc: failed to generate argument " << i << "\n";
      return nullptr;
    }
    argValues.push_back(v);
  }

  // stdlib functions
  mlir::Value stdlibResult;
  if (tryEmitStdlibCall(builder, loc, module, calleeName, argValues,
                        stdlibResult))
    return stdlibResult;

  // user-defined / unknown function
  llvm::SmallVector<mlir::Type, 4> argTypes;
  for (auto &v : argValues)
    argTypes.push_back(v.getType());

  clang::QualType returnType = calleeDecl->getReturnType();
  mlir::Type mlirReturnType = convertType(returnType);

  llvm::SmallVector<mlir::Type, 1> returnTypes;
  if (!mlir::isa<mlir::NoneType>(mlirReturnType))
    returnTypes.push_back(mlirReturnType);

  auto funcType = builder.getFunctionType(argTypes, returnTypes);
  auto funcDecl =
      getOrCreateFunctionDecl(builder, module, calleeName, funcType);

  llvm::SmallVector<mlir::Value, 4> castArgs;
  auto declaredFuncType = funcDecl.getFunctionType();
  for (auto [argVal, declaredType] :
       llvm::zip(argValues, declaredFuncType.getInputs())) {
    mlir::Type actualType = argVal.getType();
    if (actualType == declaredType) {
      castArgs.push_back(argVal);
      continue;
    }
    auto actualMemref = mlir::dyn_cast<mlir::MemRefType>(actualType);
    auto declaredMemref = mlir::dyn_cast<mlir::MemRefType>(declaredType);
    if (actualMemref && declaredMemref &&
        actualMemref.getElementType() == declaredMemref.getElementType() &&
        mlir::memref::CastOp::areCastCompatible(actualType, declaredType)) {
      auto cast =
          mlir::memref::CastOp::create(builder, loc, declaredType, argVal);
      castArgs.push_back(cast.getResult());
      continue;
    }
    castArgs.push_back(argVal);
  }

  auto callOp = mlir::func::CallOp::create(builder, loc, calleeName,
                                           returnTypes, castArgs);
  return callOp.getNumResults() > 0 ? callOp.getResult(0) : nullptr;
}

} // namespace cmlirc
