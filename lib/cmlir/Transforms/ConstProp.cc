#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#define GEN_PASS_DEF_CONSTPROPPASS
#include "cmlir/Transforms/Passes.h.inc"

namespace cmlir {

static auto foldConstantCast(mlir::Attribute attr, mlir::Type targetType,
                             mlir::Operation *op,
                             mlir::PatternRewriter &rewriter)
    -> mlir::Attribute {
  bool isTargetIntOrIndex =
      mlir::isa<mlir::IntegerType, mlir::IndexType>(targetType);
  bool isTargetFloat = mlir::isa<mlir::FloatType>(targetType);

  if (auto floatAttr = mlir::dyn_cast<mlir::FloatAttr>(attr)) {
    double val = floatAttr.getValueAsDouble();

    if (isTargetFloat) {
      return rewriter.getFloatAttr(targetType, val);
    }
    if (isTargetIntOrIndex) {
      return rewriter.getIntegerAttr(targetType, static_cast<int64_t>(val));
    }
  } else if (auto intAttr = mlir::dyn_cast<mlir::IntegerAttr>(attr)) {
    int64_t val = intAttr.getInt();

    if (isTargetIntOrIndex) {
      if (mlir::isa<mlir::arith::ExtUIOp>(op)) {
        uint64_t uval = intAttr.getValue().getZExtValue();
        return rewriter.getIntegerAttr(targetType, uval);
      }
      return rewriter.getIntegerAttr(targetType, val);
    }

    if (isTargetFloat) {
      if (mlir::isa<mlir::arith::UIToFPOp>(op)) {
        uint64_t uval = intAttr.getValue().getZExtValue();
        return rewriter.getFloatAttr(targetType, static_cast<double>(uval));
      }
      return rewriter.getFloatAttr(targetType, static_cast<double>(val));
    }
  }

  return {}; // Returns a null TypedAttr
}

template <typename CastOp>
struct FoldConstantCastPattern : public mlir::OpRewritePattern<CastOp> {
  using mlir::OpRewritePattern<CastOp>::OpRewritePattern;

  auto matchAndRewrite(CastOp castOp, mlir::PatternRewriter &rewriter) const
      -> mlir::LogicalResult override {
    mlir::Value inVal = castOp->getOperand(0);
    auto constantOp = inVal.getDefiningOp<mlir::arith::ConstantOp>();
    if (!constantOp) {
      return mlir::failure();
    }

    mlir::Attribute constantAttr = constantOp.getValue();
    mlir::Type targetType = castOp.getType();

    mlir::Attribute newAttr =
        foldConstantCast(constantAttr, targetType, castOp, rewriter);
    if (!newAttr) {
      return mlir::failure();
    }

    auto typedAttr = mlir::dyn_cast<mlir::TypedAttr>(newAttr);
    if (!typedAttr) {
      return mlir::failure();
    }

    rewriter.replaceOpWithNewOp<mlir::arith::ConstantOp>(castOp, typedAttr);

    return mlir::success();
  }
};

struct ConstPropPass : public impl::ConstPropPassBase<ConstPropPass> {
  void runOnOperation() override {
    auto op = getOperation();
    mlir::RewritePatternSet patterns(op->getContext());

    patterns.add<FoldConstantCastPattern<mlir::arith::TruncFOp>,
                 FoldConstantCastPattern<mlir::arith::ExtFOp>,
                 FoldConstantCastPattern<mlir::arith::TruncIOp>,
                 FoldConstantCastPattern<mlir::arith::ExtSIOp>,
                 FoldConstantCastPattern<mlir::arith::ExtUIOp>,
                 FoldConstantCastPattern<mlir::arith::FPToSIOp>,
                 FoldConstantCastPattern<mlir::arith::FPToUIOp>,
                 FoldConstantCastPattern<mlir::arith::SIToFPOp>,
                 FoldConstantCastPattern<mlir::arith::UIToFPOp>,
                 FoldConstantCastPattern<mlir::arith::IndexCastOp>,
                 FoldConstantCastPattern<mlir::arith::IndexCastUIOp>>(
        &getContext());

    if (mlir::failed(mlir::applyPatternsGreedily(op, std::move(patterns)))) {
      signalPassFailure();
      return;
    }
  }
};

auto createConstPropPass() -> std::unique_ptr<mlir::Pass> {
  return std::make_unique<ConstPropPass>();
}

} // namespace cmlir
