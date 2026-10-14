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
    value = clang_mlir::toValue(builder, loc, value, targetType, true);
  }

  return true;
}

auto ASTLowering::generateCallExpr(clang::CallExpr *callExpr) -> mlir::Value {
  mlir::ModuleOp module = context.module();
  mlir::OpBuilder &builder = context.builder();
  mlir::Location loc = context.source().location(callExpr->getExprLoc());

  const clang::FunctionDecl *calleeDecl = callExpr->getDirectCallee();
  if (!calleeDecl) {
    context.source().error(callExpr->getExprLoc(),
                           "indirect calls are unsupported");
    return nullptr;
  }

  if (auto *method = llvm::dyn_cast<clang::CXXMethodDecl>(calleeDecl)) {
    if (method->isVirtual() || llvm::isa<clang::CXXDestructorDecl>(method)) {
      context.source().error(
          callExpr->getExprLoc(),
          "virtual calls and destructor calls are unsupported");
      return {};
    }
  }
  std::string calleeName = functionSymbolName(*calleeDecl);
  uint32_t numArgs = callExpr->getNumArgs();

#define REGISTER_FLOAT_BYPASS(OpClass, names)                                  \
  if (matchCall(calleeName, names)) {                                          \
    std::vector<mlir::Value> args;                                             \
    for (uint32_t i = 0; i < numArgs; ++i) {                                   \
      mlir::Value arg = generateExpr(callExpr->getArg(i));                     \
      if (!arg) {                                                              \
        return {};                                                             \
      }                                                                        \
      args.push_back(arg);                                                     \
    }                                                                          \
    if (!alignArithmeticArgs(builder, loc, args, true)) {                      \
      llvm::WithColor::error()                                                 \
          << "clang-to-mlir: invalid argument type for math call '"            \
          << calleeName << "'\n";                                              \
      return nullptr;                                                          \
    }                                                                          \
    return OpClass::create(builder, loc, args).getResult();                    \
  }

#define REGISTER_INT_BYPASS(OpClass, names)                                    \
  if (matchCall(calleeName, names)) {                                          \
    std::vector<mlir::Value> args;                                             \
    for (uint32_t i = 0; i < numArgs; ++i) {                                   \
      mlir::Value arg = generateExpr(callExpr->getArg(i));                     \
      if (!arg) {                                                              \
        return {};                                                             \
      }                                                                        \
      args.push_back(arg);                                                     \
    }                                                                          \
    if (!alignArithmeticArgs(builder, loc, args, false)) {                     \
      llvm::WithColor::error()                                                 \
          << "clang-to-mlir: invalid argument type for math call '"            \
          << calleeName << "'\n";                                              \
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
      if (!arg) {                                                              \
        return {};                                                             \
      }                                                                        \
      if (mlir::isa<mlir::FloatType>(arg.getType())) {                         \
        hasFloat = true;                                                       \
      }                                                                        \
      args.push_back(arg);                                                     \
    }                                                                          \
    if (!alignArithmeticArgs(builder, loc, args, hasFloat)) {                  \
      llvm::WithColor::error()                                                 \
          << "clang-to-mlir: invalid argument type for math call '"            \
          << calleeName << "'\n";                                              \
      return nullptr;                                                          \
    }                                                                          \
    if (hasFloat) {                                                            \
      return FloatOpClass::create(builder, loc, args).getResult();             \
    }                                                                          \
    return IntOpClass::create(builder, loc, args).getResult();                 \
  }

  if (!calleeDecl->hasBody()) {
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
    REGISTER_FLOAT_BYPASS(mlir::math::CopySignOp,
                          "copysign|copysignf|copysignl")
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
    REGISTER_FLOAT_BYPASS(mlir::math::IsFiniteOp,
                          "isfinite|isfinitef|isfinitel")
    REGISTER_FLOAT_BYPASS(mlir::math::IsInfOp, "isinf|isinff|isinfl")
    REGISTER_FLOAT_BYPASS(mlir::math::IsNaNOp, "isnan|isnanf|isnanl")
    REGISTER_FLOAT_BYPASS(mlir::math::IsNormalOp,
                          "isnormal|isnormalf|isnormall")
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
  }

#undef REGISTER_FLOAT_BYPASS
#undef REGISTER_INT_BYPASS
#undef REGISTER_OVERLOAD_BYPASS

  llvm::SmallVector<CallArg, 8> argValues;

  if (auto *memberCall = llvm::dyn_cast<clang::CXXMemberCallExpr>(callExpr)) {
    if (auto *methodDecl = llvm::dyn_cast<clang::CXXMethodDecl>(calleeDecl)) {
      if (!methodDecl->isStatic()) {
        clang::Expr *implicitObj = memberCall->getImplicitObjectArgument();
        mlir::Value thisVal = generateExpr(implicitObj);
        if (!thisVal) {
          return {};
        }

        if (!mlir::isa<mlir::LLVM::LLVMPointerType>(thisVal.getType())) {
          mlir::Type structType = thisVal.getType();
          mlir::Value one =
              clang_mlir::intConst(builder, loc, builder.getI32Type(), 1);
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
    if (!value) {
      return {};
    }

    argValues.push_back({value, argExpr->getType()});
  }

  mlir::Value stdlibResult;
  if (!calleeDecl->hasBody() &&
      tryEmitStdlibCall(builder, loc, module, calleeName, argValues,
                        stdlibResult)) {
    return stdlibResult;
  }

  // Definitions use the internal MLIR signature. Declarations without a body
  // cross the external ABI and must receive raw pointers, not memref
  // descriptors.
  if (!calleeDecl->hasBody()) {
    llvm::SmallVector<mlir::Type, 8> fixedTypes;
    for (const auto *parameter : calleeDecl->parameters()) {
      auto type = types.convertStorageType(parameter->getType());
      if (!type || (!type.isIntOrFloat() &&
                    !mlir::isa<mlir::LLVM::LLVMPointerType>(type))) {
        context.source().error(callExpr->getExprLoc(),
                               "unsupported external ABI parameter");
        return {};
      }
      fixedTypes.push_back(type);
    }
    auto resultType = types.convertStorageType(calleeDecl->getReturnType());
    if (mlir::isa_and_nonnull<mlir::NoneType>(resultType)) {
      resultType = mlir::LLVM::LLVMVoidType::get(builder.getContext());
    }
    if (!resultType ||
        (!resultType.isIntOrFloat() &&
         !mlir::isa<mlir::LLVM::LLVMPointerType, mlir::LLVM::LLVMVoidType>(
             resultType))) {
      context.source().error(callExpr->getExprLoc(),
                             "unsupported external ABI result");
      return {};
    }
    mlir::Value result;
    emitExternalCall(builder, loc, module, calleeName, argValues, fixedTypes,
                     resultType, calleeDecl->isVariadic(), result);
    return result;
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

  if (!funcDecl) {
    return {};
  }
  llvm::SmallVector<mlir::Value, 8> castArgs;
  auto declaredFuncType = funcDecl.getFunctionType();

  if (declaredFuncType.getNumInputs() != argValues.size()) {
    llvm::WithColor::error()
        << "clang-to-mlir: function '" << calleeName << "' expects "
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

} // namespace cmlir::frontend
