#include "../../Converter.h"
#include "../Types/Types.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include <string>

namespace cmlirc {

std::vector<std::string> split_string(const std::string &str, char delimiter) {
  std::vector<std::string> tokens;
  std::string token;
  std::istringstream tokenStream(str);
  while (std::getline(tokenStream, token, delimiter)) {
    tokens.push_back(token);
  }
  return tokens;
}

mlir::func::FuncOp getOrCreateFunctionDecl(mlir::OpBuilder &builder,
                                           mlir::ModuleOp module,
                                           const std::string &name,
                                           mlir::FunctionType funcType) {
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

mlir::Value CMLIRConverter::generateCallExpr(clang::CallExpr *callExpr) {
  mlir::OpBuilder &builder = context_manager_.Builder();
  mlir::Location loc = builder.getUnknownLoc();

  const clang::FunctionDecl *calleeDecl = callExpr->getDirectCallee();
  if (!calleeDecl) {
    llvm::errs() << "Indirect calls not supported\n";
    return nullptr;
  }

  std::string calleeName = calleeDecl->getNameAsString();
  uint32_t num_args = callExpr->getNumArgs();
  std::vector<std::string> tokens;
  std::string token;

#define REGISTER_MATH_CALL(op, names)                                          \
  {                                                                            \
    std::istringstream tokenStream(names);                                     \
                                                                               \
    while (std::getline(tokenStream, token, '|')) {                            \
      tokens.push_back(token);                                                 \
    }                                                                          \
                                                                               \
    for (const auto &token : tokens) {                                         \
      if (calleeName == token) {                                               \
        std::vector<mlir::Value> args;                                         \
        for (uint32_t i = 0; i < num_args; ++i) {                              \
          mlir::Value arg = generateExpr(callExpr->getArg(i));                 \
          if (!arg) {                                                          \
            llvm::errs() << "Failed to generate argument " << i << "\n";       \
            return nullptr;                                                    \
          }                                                                    \
          args.push_back(arg);                                                 \
        }                                                                      \
        return mlir::math::op::create(builder, loc, args);                     \
      }                                                                        \
    }                                                                          \
  }

  REGISTER_MATH_CALL(AbsFOp, "fabs|fabsf|fabsl")
  REGISTER_MATH_CALL(AbsIOp, "abs|absf|absl")
  REGISTER_MATH_CALL(AcosOp, "acos|acosf|acosl")
  REGISTER_MATH_CALL(AcoshOp, "acosh|acoshf|acoshl")
  REGISTER_MATH_CALL(AsinOp, "asin|asinf|asinl")
  REGISTER_MATH_CALL(AsinhOp, "asinh|asinhf|asinhl")
  REGISTER_MATH_CALL(AtanOp, "atan|atanf|atanl")
  REGISTER_MATH_CALL(Atan2Op, "atan2|atan2f|atan2l")
  REGISTER_MATH_CALL(AtanhOp, "atanh|atanhf|atanhl")
  REGISTER_MATH_CALL(CbrtOp, "cbrt|cbrtf|cbrtl")
  REGISTER_MATH_CALL(CeilOp, "ceil|ceilf|ceill")
  REGISTER_MATH_CALL(ClampFOp, "clamp|clampf|clampl")
  REGISTER_MATH_CALL(CopySignOp, "copysign|copysignf|copysignl")
  REGISTER_MATH_CALL(CosOp, "cos|cosf|cosl")
  REGISTER_MATH_CALL(CoshOp, "cosh|coshf|coshl")
  REGISTER_MATH_CALL(CountLeadingZerosOp, "ctlz|ctlzf|ctlzl")
  REGISTER_MATH_CALL(CtPopOp, "ctpop|ctpopf|ctpopl")
  REGISTER_MATH_CALL(CountTrailingZerosOp, "cttz|cttzf|cttzl")
  REGISTER_MATH_CALL(ErfOp, "erf|erff|erfl")
  REGISTER_MATH_CALL(ErfcOp, "erfc|erfcf|erfcl")
  REGISTER_MATH_CALL(ExpOp, "exp|expf|expl")
  REGISTER_MATH_CALL(Exp2Op, "exp2|exp2f|exp2l")
  REGISTER_MATH_CALL(ExpM1Op, "expm1|expm1f|expm1l")
  REGISTER_MATH_CALL(FloorOp, "floor|floorf|floorl")
  REGISTER_MATH_CALL(FmaOp, "fma|fmaf|fmal")
  REGISTER_MATH_CALL(FPowIOp, "fpowi|fpowif|fpowil")
  REGISTER_MATH_CALL(IPowIOp, "ipowi|ipowif|ipowil")
  REGISTER_MATH_CALL(IsFiniteOp, "isfinite|isfinitef|isfinitel")
  REGISTER_MATH_CALL(IsInfOp, "isinf|isinff|isinfl")
  REGISTER_MATH_CALL(IsNaNOp, "isnan|isnanf|isnanl")
  REGISTER_MATH_CALL(IsNormalOp, "isnormal|isnormalf|isnormall")
  REGISTER_MATH_CALL(LogOp, "log|logf|logl")
  REGISTER_MATH_CALL(Log10Op, "log10|log10f|log10l")
  REGISTER_MATH_CALL(Log1pOp, "log1p|log1pf|log1pl")
  REGISTER_MATH_CALL(Log2Op, "log2|log2f|log2l")
  REGISTER_MATH_CALL(PowFOp, "pow|powf|powl")
  REGISTER_MATH_CALL(RoundOp, "round|roundf|roundl")
  REGISTER_MATH_CALL(RoundEvenOp, "roundeven|roundevenf|roundevenl")
  REGISTER_MATH_CALL(RsqrtOp, "rsqrt|rsqrtf|rsqrtl")
  REGISTER_MATH_CALL(SinOp, "sin|sinf|sinl")
  REGISTER_MATH_CALL(SinhOp, "sinh|sinhf|sinhl")
  REGISTER_MATH_CALL(SqrtOp, "sqrt|sqrtf|sqrtl")
  REGISTER_MATH_CALL(TanOp, "tan|tanf|tanl")
  REGISTER_MATH_CALL(TanhOp, "tanh|tanhf|tanhl")
  REGISTER_MATH_CALL(TruncOp, "trunc|truncf|truncl")

#undef REGISTER_MATH_CALL

  llvm::SmallVector<mlir::Value, 4> argValues;
  llvm::SmallVector<mlir::Type, 4> argTypes;

  for (unsigned i = 0; i < callExpr->getNumArgs(); ++i) {
    mlir::Value argValue = generateExpr(callExpr->getArg(i));
    if (!argValue) {
      llvm::errs() << "Failed to generate argument " << i << "\n";
      return nullptr;
    }
    argValues.push_back(argValue);
    argTypes.push_back(argValue.getType());
  }

  clang::QualType returnType = calleeDecl->getReturnType();
  mlir::Type mlirReturnType = convertType(builder, returnType);

  llvm::SmallVector<mlir::Type, 1> returnTypes;
  if (!mlir::isa<mlir::NoneType>(mlirReturnType)) {
    returnTypes.push_back(mlirReturnType);
  }

  auto funcType = builder.getFunctionType(argTypes, returnTypes);
  mlir::ModuleOp module = context_manager_.Module();
  getOrCreateFunctionDecl(builder, module, calleeName, funcType);

  auto callOp = mlir::func::CallOp::create(builder, loc, calleeName,
                                           returnTypes, argValues);

  return callOp.getNumResults() > 0 ? callOp.getResult(0) : nullptr;
}

} // namespace cmlirc
