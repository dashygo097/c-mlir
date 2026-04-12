#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#define GEN_PASS_DEF_ARITHCASTPROPPASS
#include "cmlir/Transforms/Passes.h.inc"

namespace cmlir {

static auto getIntVal(mlir::Value floatVal, mlir::Type intType,
                      mlir::PatternRewriter &rewriter, mlir::Location loc)
    -> mlir::Value {
  if (auto sitofp = floatVal.getDefiningOp<mlir::arith::SIToFPOp>()) {
    if (sitofp.getIn().getType() == intType) {
      return sitofp.getIn();
    }
  }

  llvm::APFloat floatAttr(0.0);
  if (mlir::matchPattern(floatVal, mlir::m_ConstantFloat(&floatAttr))) {
    double dVal = floatAttr.convertToDouble();
    auto intVal = static_cast<int64_t>(dVal);
    return mlir::arith::ConstantOp::create(
        rewriter, loc, rewriter.getIntegerAttr(intType, intVal));
  }

  return nullptr;
}

template <typename FloatOp, typename IntOp>
static auto tryReplaceBinary(mlir::arith::FPToSIOp fptosiOp,
                             mlir::Value inFloat, mlir::Type outIntType,
                             mlir::PatternRewriter &rewriter)
    -> mlir::LogicalResult {
  if (auto binOp = inFloat.getDefiningOp<FloatOp>()) {
    mlir::Value lhsInt =
        getIntVal(binOp.getLhs(), outIntType, rewriter, fptosiOp.getLoc());
    mlir::Value rhsInt =
        getIntVal(binOp.getRhs(), outIntType, rewriter, fptosiOp.getLoc());

    if (lhsInt && rhsInt) {
      rewriter.replaceOpWithNewOp<IntOp>(fptosiOp, lhsInt, rhsInt);
      return mlir::success();
    }
  }
  return mlir::failure();
}

struct OptimizeFPToSIPattern
    : public mlir::OpRewritePattern<mlir::arith::FPToSIOp> {
  using mlir::OpRewritePattern<mlir::arith::FPToSIOp>::OpRewritePattern;

  auto matchAndRewrite(mlir::arith::FPToSIOp fptosiOp,
                       mlir::PatternRewriter &rewriter) const
      -> mlir::LogicalResult override {
    mlir::Value inFloat = fptosiOp.getIn();
    mlir::Type outIntType = fptosiOp.getType();

    // Map MaxNumF -> MaxSI
    if (mlir::succeeded(
            tryReplaceBinary<mlir::arith::MaxNumFOp, mlir::arith::MaxSIOp>(
                fptosiOp, inFloat, outIntType, rewriter))) {
      return mlir::success();
    }

    // Map MinNumF -> MinSI
    if (mlir::succeeded(
            tryReplaceBinary<mlir::arith::MinNumFOp, mlir::arith::MinSIOp>(
                fptosiOp, inFloat, outIntType, rewriter))) {
      return mlir::success();
    }

    // Map AddF -> AddI
    if (mlir::succeeded(
            tryReplaceBinary<mlir::arith::AddFOp, mlir::arith::AddIOp>(
                fptosiOp, inFloat, outIntType, rewriter))) {
      return mlir::success();
    }

    // Map SubF -> SubI
    if (mlir::succeeded(
            tryReplaceBinary<mlir::arith::SubFOp, mlir::arith::SubIOp>(
                fptosiOp, inFloat, outIntType, rewriter))) {
      return mlir::success();
    }

    // Map MulF -> MulI
    if (mlir::succeeded(
            tryReplaceBinary<mlir::arith::MulFOp, mlir::arith::MulIOp>(
                fptosiOp, inFloat, outIntType, rewriter))) {
      return mlir::success();
    }

    return mlir::failure();
  }
};

struct ArithCastPropPass
    : public impl::ArithCastPropPassBase<ArithCastPropPass> {
  void runOnOperation() override {
    auto op = getOperation();
    mlir::RewritePatternSet patterns(op->getContext());

    patterns.add<OptimizeFPToSIPattern>(&getContext());

    if (mlir::failed(mlir::applyPatternsGreedily(op, std::move(patterns)))) {
      signalPassFailure();
      return;
    }
  }
};

auto createArithCastPropPass() -> std::unique_ptr<mlir::Pass> {
  return std::make_unique<ArithCastPropPass>();
}

} // namespace cmlir
