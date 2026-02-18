#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Dominance.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#define GEN_PASS_DEF_STRUCT2MEMREF
#include "cmlir/Transforms/Passes.h.inc"

namespace cmlir {

struct Struct2MemrefPattern
    : public mlir::OpRewritePattern<mlir::func::CallOp> {
  using mlir::OpRewritePattern<mlir::func::CallOp>::OpRewritePattern;
  mlir::LogicalResult
  matchAndRewrite(mlir::func::CallOp call,
                  mlir::PatternRewriter &rewriter) const override {
    return mlir::failure();
  }
};

struct Struct2MemrefPass
    : public mlir::PassWrapper<Struct2MemrefPass,
                               mlir::OperationPass<mlir::func::FuncOp>> {

  void runOnOperation() override {
    mlir::func::FuncOp func = getOperation();
    mlir::RewritePatternSet patterns(&getContext());

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

std::unique_ptr<mlir::Pass> createStruct2MemrefPass() {
  return std::make_unique<Struct2MemrefPass>();
}

} // namespace cmlir
