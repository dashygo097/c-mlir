#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Dominance.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#define GEN_PASS_DEF_CONSTPROPPASS
#include "cmlir/Transforms/Passes.h.inc"

namespace cmlir {

struct FoldConstantCastIntoAffineStore
    : public mlir::OpRewritePattern<mlir::affine::AffineStoreOp> {
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::affine::AffineStoreOp storeOp,
                  mlir::PatternRewriter &rewriter) const override {
    mlir::Value storedValue = storeOp.getValue();

    mlir::Operation *defOp = storedValue.getDefiningOp();
    if (!defOp)
      return mlir::failure();

    mlir::Value constantValue;
    mlir::Type targetType = storedValue.getType();

    if (auto truncFOp = mlir::dyn_cast<mlir::arith::TruncFOp>(defOp)) {
      // arith.truncf : f64 -> f32
      constantValue = truncFOp.getIn();
    } else if (auto extFOp = mlir::dyn_cast<mlir::arith::ExtFOp>(defOp)) {
      // arith.extf : f32 -> f64
      constantValue = extFOp.getIn();
    } else if (auto truncIOp = mlir::dyn_cast<mlir::arith::TruncIOp>(defOp)) {
      // arith.trunci : i64 -> i32
      constantValue = truncIOp.getIn();
    } else if (auto extSIOp = mlir::dyn_cast<mlir::arith::ExtSIOp>(defOp)) {
      // arith.extsi : i32 -> i64
      constantValue = extSIOp.getIn();
    } else if (auto extUIOp = mlir::dyn_cast<mlir::arith::ExtUIOp>(defOp)) {
      // arith.extui : i32 -> i64
      constantValue = extUIOp.getIn();
    } else if (auto fpToSIOp = mlir::dyn_cast<mlir::arith::FPToSIOp>(defOp)) {
      // arith.fptosi : f32 -> i32
      constantValue = fpToSIOp.getIn();
    } else if (auto siToFPOp = mlir::dyn_cast<mlir::arith::SIToFPOp>(defOp)) {
      // arith.sitofp : i32 -> f32
      constantValue = siToFPOp.getIn();
    } else {
      return mlir::failure();
    }

    auto constantOp = constantValue.getDefiningOp<mlir::arith::ConstantOp>();
    if (!constantOp)
      return mlir::failure();

    mlir::Attribute constantAttr = constantOp.getValue();

    mlir::Attribute newConstantAttr;

    if (auto floatAttr = mlir::dyn_cast<mlir::FloatAttr>(constantAttr)) {
      if (auto targetFloatType = mlir::dyn_cast<mlir::FloatType>(targetType)) {
        double value = floatAttr.getValueAsDouble();
        newConstantAttr = rewriter.getFloatAttr(targetFloatType, value);
      } else if (auto targetIntType =
                     mlir::dyn_cast<mlir::IntegerType>(targetType)) {
        // float -> int
        int64_t value = static_cast<int64_t>(floatAttr.getValueAsDouble());
        newConstantAttr = rewriter.getIntegerAttr(targetIntType, value);
      } else {
        return mlir::failure();
      }
    } else if (auto intAttr = mlir::dyn_cast<mlir::IntegerAttr>(constantAttr)) {
      if (auto targetIntType = mlir::dyn_cast<mlir::IntegerType>(targetType)) {
        int64_t value = intAttr.getInt();
        newConstantAttr = rewriter.getIntegerAttr(targetIntType, value);
      } else if (auto targetFloatType =
                     mlir::dyn_cast<mlir::FloatType>(targetType)) {
        // int -> float
        double value = static_cast<double>(intAttr.getInt());
        newConstantAttr = rewriter.getFloatAttr(targetFloatType, value);
      } else {
        return mlir::failure();
      }
    } else {
      return mlir::failure();
    }

    auto newConstant = mlir::arith::ConstantOp::materialize(
        rewriter, newConstantAttr, targetType, storeOp.getLoc());

    if (!newConstant)
      return mlir::failure();

    rewriter.replaceOpWithNewOp<mlir::affine::AffineStoreOp>(
        storeOp, newConstant.getResult(), storeOp.getMemRef(),
        storeOp.getAffineMap(), storeOp.getIndices());

    return mlir::success();
  }
};

struct FoldConstantCastIntoMemRefStore
    : public mlir::OpRewritePattern<mlir::memref::StoreOp> {
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::memref::StoreOp storeOp,
                  mlir::PatternRewriter &rewriter) const override {
    mlir::Value storedValue = storeOp.getValue();

    mlir::Operation *defOp = storedValue.getDefiningOp();
    if (!defOp)
      return mlir::failure();

    mlir::Value constantValue;
    mlir::Type targetType = storedValue.getType();

    if (auto truncFOp = mlir::dyn_cast<mlir::arith::TruncFOp>(defOp)) {
      constantValue = truncFOp.getIn();
    } else if (auto extFOp = mlir::dyn_cast<mlir::arith::ExtFOp>(defOp)) {
      constantValue = extFOp.getIn();
    } else if (auto truncIOp = mlir::dyn_cast<mlir::arith::TruncIOp>(defOp)) {
      constantValue = truncIOp.getIn();
    } else if (auto extSIOp = mlir::dyn_cast<mlir::arith::ExtSIOp>(defOp)) {
      constantValue = extSIOp.getIn();
    } else if (auto extUIOp = mlir::dyn_cast<mlir::arith::ExtUIOp>(defOp)) {
      constantValue = extUIOp.getIn();
    } else {
      return mlir::failure();
    }

    auto constantOp = constantValue.getDefiningOp<mlir::arith::ConstantOp>();
    if (!constantOp)
      return mlir::failure();

    mlir::Attribute constantAttr = constantOp.getValue();
    mlir::Attribute newConstantAttr;

    if (auto floatAttr = mlir::dyn_cast<mlir::FloatAttr>(constantAttr)) {
      if (auto targetFloatType = mlir::dyn_cast<mlir::FloatType>(targetType)) {
        double value = floatAttr.getValueAsDouble();
        newConstantAttr = rewriter.getFloatAttr(targetFloatType, value);
      } else {
        return mlir::failure();
      }
    } else if (auto intAttr = mlir::dyn_cast<mlir::IntegerAttr>(constantAttr)) {
      if (auto targetIntType = mlir::dyn_cast<mlir::IntegerType>(targetType)) {
        int64_t value = intAttr.getInt();
        newConstantAttr = rewriter.getIntegerAttr(targetIntType, value);
      } else {
        return mlir::failure();
      }
    } else {
      return mlir::failure();
    }

    auto newConstant = mlir::arith::ConstantOp::materialize(
        rewriter, newConstantAttr, targetType, storeOp.getLoc());

    rewriter.replaceOpWithNewOp<mlir::memref::StoreOp>(
        storeOp, newConstant.getResult(), storeOp.getMemRef(),
        storeOp.getIndices());

    return mlir::success();
  }
};

struct ConstPropPass
    : public mlir::PassWrapper<ConstPropPass,
                               mlir::OperationPass<mlir::func::FuncOp>> {

  void runOnOperation() override {
    mlir::func::FuncOp func = getOperation();
    mlir::RewritePatternSet patterns(&getContext());

    patterns.add<FoldConstantCastIntoAffineStore>(&getContext());
    patterns.add<FoldConstantCastIntoMemRefStore>(&getContext());

    if (mlir::failed(mlir::applyPatternsGreedily(func, std::move(patterns)))) {
      signalPassFailure();
      return;
    }

    func.walk([](mlir::Operation *op) {
      if (mlir::isOpTriviallyDead(op)) {
        op->erase();
      }
    });
  }
};

std::unique_ptr<mlir::Pass> createConstPropPass() {
  return std::make_unique<ConstPropPass>();
}

} // namespace cmlir
