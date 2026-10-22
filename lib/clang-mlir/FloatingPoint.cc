#include "clang-mlir/FloatingPoint.h"

namespace clang_mlir {

auto getFastMathFlags(const clang::Expr &expr, const clang::LangOptions &lang)
    -> mlir::arith::FastMathFlags {
  auto options = expr.getFPFeaturesInEffect(lang);
  using Flags = mlir::arith::FastMathFlags;
  auto flags = Flags::none;
  if (options.getAllowFPReassociate()) {
    flags |= Flags::reassoc;
  }
  if (options.getNoHonorNaNs()) {
    flags |= Flags::nnan;
  }
  if (options.getNoHonorInfs()) {
    flags |= Flags::ninf;
  }
  if (options.getNoSignedZero()) {
    flags |= Flags::nsz;
  }
  if (options.getAllowReciprocal()) {
    flags |= Flags::arcp;
  }
  if (options.getAllowApproxFunc()) {
    flags |= Flags::afn;
  }
  // contract(on) only allows fusion within an expression. Marking every op
  // would incorrectly permit fusion across separate source statements.
  if (options.allowFPContractAcrossStatement()) {
    flags |= Flags::contract;
  }
  return flags;
}

} // namespace clang_mlir
