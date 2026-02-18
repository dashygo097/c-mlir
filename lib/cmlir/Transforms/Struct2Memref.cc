#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
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

struct ConvertFuncSignatureStruct2MemrefPattern
    : public mlir::OpRewritePattern<mlir::func::FuncOp> {
  using mlir::OpRewritePattern<mlir::func::FuncOp>::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::func::FuncOp funcOp,
                  mlir::PatternRewriter &rewriter) const override {
    auto funcType = funcOp.getFunctionType();
    bool needConversion = false;

    // check if any inputs have LLVM struct types
    for (const mlir::Type &type : funcType.getInputs()) {
      if (mlir::isa<mlir::LLVM::LLVMStructType>(type)) {
        needConversion = true;
        break;
      }
    }

    // check if any results have LLVM struct types
    if (!needConversion) {
      for (const mlir::Type &type : funcType.getResults()) {
        if (mlir::isa<mlir::LLVM::LLVMStructType>(type)) {
          needConversion = true;
          break;
        }
      }
    }

    if (!needConversion) {
      return mlir::failure();
    }

    // convert input types
    llvm::SmallVector<mlir::Type, 4> newInputTypes;
    for (const mlir::Type &type : funcType.getInputs()) {
      if (auto structType = mlir::dyn_cast<mlir::LLVM::LLVMStructType>(type)) {
        auto memrefType =
            convertStructToMemref(structType, funcOp.getContext());
        if (!memrefType) {
          return mlir::failure();
        }
        newInputTypes.push_back(memrefType);
      } else {
        newInputTypes.push_back(type);
      }
    }

    // convert result types
    for (const mlir::Type &type : funcType.getResults()) {
      if (auto structType = mlir::dyn_cast<mlir::LLVM::LLVMStructType>(type)) {
        auto memrefType =
            convertStructToMemref(structType, funcOp.getContext());
        if (!memrefType) {
          return mlir::failure();
        }
        newInputTypes.push_back(memrefType);
      } else {
        newInputTypes.push_back(type);
      }
    }

    auto newFuncType = mlir::FunctionType::get(
        funcOp.getContext(), newInputTypes, funcType.getResults());

    funcOp.setFunctionTypeAttr(mlir::TypeAttr::get(newFuncType));

    mlir::Block &entryBlock = funcOp.getBody().front();
    for (mlir::BlockArgument &arg : entryBlock.getArguments()) {
      if (auto structType =
              mlir::dyn_cast<mlir::LLVM::LLVMStructType>(arg.getType())) {
        auto memrefType =
            convertStructToMemref(structType, funcOp.getContext());
        if (!memrefType) {
          return mlir::failure();
        }
        arg.setType(memrefType);
      }
    }

    return mlir::success();
  }
};

struct ConvertStructAlloca2Memref
    : public mlir::OpRewritePattern<mlir::LLVM::AllocaOp> {
  using mlir::OpRewritePattern<mlir::LLVM::AllocaOp>::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::LLVM::AllocaOp allocaOp,
                  mlir::PatternRewriter &rewriter) const override {}
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
