#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Dominance.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#define GEN_PASS_DEF_STRUCT2MEMREF
#include "cmlir/Transforms/Passes.h.inc"

namespace cmlir {

static mlir::Type convertStructToMemref(mlir::LLVM::LLVMStructType structType,
                                        mlir::MLIRContext *mlirCtx) {
  if (!structType)
    return nullptr;

  auto body = structType.getBody();
  if (body.empty())
    return nullptr;

  mlir::Type elementType = body[0];
  bool allSameType =
      llvm::all_of(body, [&](mlir::Type t) { return t == elementType; });

  if (allSameType) {
    return mlir::MemRefType::get({static_cast<int64_t>(body.size())},
                                 elementType);
  }

  return nullptr;
}

/*
 * e.g.
 * struct Point {
 *  float x;
 *  float y;
 *  } -> memref<2xf32>
 */

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
