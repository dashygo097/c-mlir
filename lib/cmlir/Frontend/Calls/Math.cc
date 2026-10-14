#include "Math.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "clang/AST/ASTContext.h"

namespace cmlir::frontend {

namespace {

template <typename Op>
auto emitMathOp(mlir::OpBuilder &builder, mlir::Location loc,
                llvm::ArrayRef<mlir::Value> arguments) -> mlir::Value {
  if constexpr (Op::hasProperties()) {
    typename Op::Properties properties;
    Op::populateDefaultProperties(
        mlir::OperationName(Op::getOperationName(), builder.getContext()),
        properties);
    return Op::create(builder, loc, mlir::ValueRange(arguments), properties)
        .getResult();
  } else {
    return Op::create(builder, loc, mlir::ValueRange(arguments)).getResult();
  }
}

template <typename Op>
auto intrinsic(llvm::StringLiteral names, unsigned arity, MathDomain domain,
               bool predicate = false) -> MathIntrinsic {
  return {names, emitMathOp<Op>, arity, domain, predicate};
}

const auto integer = MathDomain::Integer;
const auto floating = MathDomain::Float;

const MathIntrinsic intrinsics[] = {
    intrinsic<mlir::arith::MinSIOp>("min|mini|minl", 2, integer),
    intrinsic<mlir::arith::MaxSIOp>("max|maxi|maxl", 2, integer),
    intrinsic<mlir::arith::MinNumFOp>("fmin|fminf|fminl", 2, floating),
    intrinsic<mlir::arith::MaxNumFOp>("fmax|fmaxf|fmaxl", 2, floating),
    intrinsic<mlir::math::AbsIOp>("abs|absi|absl|labs|llabs", 1, integer),
    intrinsic<mlir::math::AbsFOp>("fabs|fabsf|fabsl", 1, floating),
    intrinsic<mlir::math::AcosOp>("acos|acosf|acosl", 1, floating),
    intrinsic<mlir::math::AcoshOp>("acosh|acoshf|acoshl", 1, floating),
    intrinsic<mlir::math::AsinOp>("asin|asinf|asinl", 1, floating),
    intrinsic<mlir::math::AsinhOp>("asinh|asinhf|asinhl", 1, floating),
    intrinsic<mlir::math::AtanOp>("atan|atanf|atanl", 1, floating),
    intrinsic<mlir::math::Atan2Op>("atan2|atan2f|atan2l", 2, floating),
    intrinsic<mlir::math::AtanhOp>("atanh|atanhf|atanhl", 1, floating),
    intrinsic<mlir::math::CbrtOp>("cbrt|cbrtf|cbrtl", 1, floating),
    intrinsic<mlir::math::CeilOp>("ceil|ceilf|ceill", 1, floating),
    intrinsic<mlir::math::CopySignOp>("copysign|copysignf|copysignl", 2,
                                      floating),
    intrinsic<mlir::math::CosOp>("cos|cosf|cosl", 1, floating),
    intrinsic<mlir::math::CoshOp>("cosh|coshf|coshl", 1, floating),
    intrinsic<mlir::math::CountLeadingZerosOp>("ctlz|ctlzl", 1, integer),
    intrinsic<mlir::math::CtPopOp>("ctpop|ctpopl", 1, integer),
    intrinsic<mlir::math::CountTrailingZerosOp>("cttz|cttzl", 1, integer),
    intrinsic<mlir::math::ErfOp>("erf|erff|erfl", 1, floating),
    intrinsic<mlir::math::ErfcOp>("erfc|erfcf|erfcl", 1, floating),
    intrinsic<mlir::math::ExpOp>("exp|expf|expl", 1, floating),
    intrinsic<mlir::math::Exp2Op>("exp2|exp2f|exp2l", 1, floating),
    intrinsic<mlir::math::ExpM1Op>("expm1|expm1f|expm1l", 1, floating),
    intrinsic<mlir::math::FloorOp>("floor|floorf|floorl", 1, floating),
    intrinsic<mlir::math::FmaOp>("fma|fmaf|fmal", 3, floating),
    intrinsic<mlir::math::IsFiniteOp>("isfinite|isfinitef|isfinitel", 1,
                                      floating, true),
    intrinsic<mlir::math::IsInfOp>("isinf|isinff|isinfl", 1, floating, true),
    intrinsic<mlir::math::IsNaNOp>("isnan|isnanf|isnanl", 1, floating, true),
    intrinsic<mlir::math::IsNormalOp>("isnormal|isnormalf|isnormall", 1,
                                      floating, true),
    intrinsic<mlir::math::LogOp>("log|logf|logl", 1, floating),
    intrinsic<mlir::math::Log10Op>("log10|log10f|log10l", 1, floating),
    intrinsic<mlir::math::Log1pOp>("log1p|log1pf|log1pl", 1, floating),
    intrinsic<mlir::math::Log2Op>("log2|log2f|log2l", 1, floating),
    intrinsic<mlir::math::PowFOp>("pow|powf|powl", 2, floating),
    intrinsic<mlir::math::RoundOp>("round|roundf|roundl", 1, floating),
    intrinsic<mlir::math::RoundEvenOp>("roundeven|roundevenf|roundevenl", 1,
                                       floating),
    intrinsic<mlir::math::RsqrtOp>("rsqrt|rsqrtf|rsqrtl", 1, floating),
    intrinsic<mlir::math::SinOp>("sin|sinf|sinl", 1, floating),
    intrinsic<mlir::math::SinhOp>("sinh|sinhf|sinhl", 1, floating),
    intrinsic<mlir::math::SqrtOp>("sqrt|sqrtf|sqrtl", 1, floating),
    intrinsic<mlir::math::TanOp>("tan|tanf|tanl", 1, floating),
    intrinsic<mlir::math::TanhOp>("tanh|tanhf|tanhl", 1, floating),
    intrinsic<mlir::math::TruncOp>("trunc|truncf|truncl", 1, floating),
};

} // namespace

auto findMathIntrinsic(const clang::FunctionDecl &decl,
                       mlir::FunctionType signature) -> const MathIntrinsic * {
  // Methods and C++ namespace functions must retain their own identity.
  if (!decl.getIdentifier() || (decl.isVariadic() && !decl.getBuiltinID()) ||
      (decl.getASTContext().getLangOpts().CPlusPlus && !decl.isExternC() &&
       !decl.getBuiltinID()) ||
      signature.getNumResults() != 1) {
    return nullptr;
  }
  auto name = decl.getName();
  name.consume_front("__builtin_");
  for (const auto &entry : intrinsics) {
    llvm::SmallVector<llvm::StringRef, 4> names;
    entry.names.split(names, '|');
    if (!llvm::is_contained(names, name) ||
        signature.getNumInputs() != entry.arity) {
      continue;
    }
    auto type = signature.getInput(0);
    if ((entry.domain == MathDomain::Float &&
         !mlir::isa<mlir::FloatType>(type)) ||
        (entry.domain == MathDomain::Integer &&
         !mlir::isa<mlir::IntegerType>(type))) {
      continue;
    }
    if (!llvm::all_of(signature.getInputs(), [&](mlir::Type input) -> bool {
          return input == type;
        })) {
      continue;
    }
    auto result = signature.getResult(0);
    if (entry.predicate ? !mlir::isa<mlir::IntegerType>(result)
                        : result != type) {
      continue;
    }
    // Signed integer min/max must not reinterpret unsigned ordering.
    if (entry.domain == MathDomain::Integer && entry.arity == 2 &&
        decl.getParamDecl(0)->getType()->isUnsignedIntegerType()) {
      continue;
    }
    return &entry;
  }
  return nullptr;
}

} // namespace cmlir::frontend
