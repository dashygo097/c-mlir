#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Dominance.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#define GEN_PASS_DEF_STRUCT2MEMREFPASS
#include "cmlir/Transforms/Passes.h.inc"

namespace cmlir {

// e.g.
// struct Point {
//  float x;
//  float y;
// }
// =>
// memref<2xf32>

static auto isStructElemSameType(mlir::LLVM::LLVMStructType structType)
    -> bool {
  if (structType.getBody().empty()) {
    return false;
  }

  return llvm::all_of(structType.getBody(), [&](mlir::Type elemType) -> bool {
    return elemType == structType.getBody()[0];
  });
}

static auto structToMemrefType(mlir::LLVM::LLVMStructType st)
    -> mlir::MemRefType {
  auto elemType = st.getBody()[0];
  return mlir::MemRefType::get(
      {mlir::ShapedType::kDynamic, static_cast<int64_t>(st.getBody().size())},
      elemType);
}

static auto inferStructTypeFromPtr(mlir::BlockArgument arg)
    -> mlir::LLVM::LLVMStructType {
  for (mlir::Operation *user : arg.getUsers()) {
    if (auto gep = mlir::dyn_cast<mlir::LLVM::GEPOp>(user)) {
      if (auto structType =
              mlir::dyn_cast<mlir::LLVM::LLVMStructType>(gep.getElemType())) {
        return structType;
      }
    }
  }
  return nullptr;
}

// func.func @foo(%arg0: !llvm.struct<(f32, f32)>) -> ...
// =>
// func.func @foo(%arg0: memref<?x2xf32>) -> ...
struct LLVMStructFuncArgToMemrefPattern
    : public mlir::OpRewritePattern<mlir::func::FuncOp> {
  using mlir::OpRewritePattern<mlir::func::FuncOp>::OpRewritePattern;

  auto matchAndRewrite(mlir::func::FuncOp funcOp,
                       mlir::PatternRewriter &rewriter) const
      -> mlir::LogicalResult override {
    mlir::FunctionType funcType = funcOp.getFunctionType();
    bool anyChanged = false;

    llvm::SmallVector<mlir::Type> newInputTypes;

    bool hasBody = !funcOp.getBody().empty();

    for (auto [idx, argType] : llvm::enumerate(funcType.getInputs())) {
      mlir::LLVM::LLVMStructType structType = nullptr;

      if (auto st = mlir::dyn_cast<mlir::LLVM::LLVMStructType>(argType)) {
        structType = st;
      } else if (hasBody && mlir::isa<mlir::LLVM::LLVMPointerType>(argType)) {
        structType = inferStructTypeFromPtr(funcOp.front().getArgument(idx));
      }

      if (structType && isStructElemSameType(structType)) {
        newInputTypes.push_back(structToMemrefType(structType));
        anyChanged = true;
      } else {
        newInputTypes.push_back(argType);
      }
    }

    if (!anyChanged) {
      return mlir::failure();
    }

    auto newFuncType =
        rewriter.getFunctionType(newInputTypes, funcType.getResults());

    rewriter.modifyOpInPlace(funcOp, [&]() -> void {
      funcOp.setType(newFuncType);
      if (hasBody) {
        mlir::Block &entryBlock = funcOp.front();
        for (auto [idx, argType] : llvm::enumerate(newInputTypes)) {
          if (argType != funcType.getInputs()[idx]) {
            entryBlock.getArgument(idx).setType(argType);
          }
        }
      }
    });

    return mlir::success();
  }
};

// %ptr = llvm.alloca %structType, %arraySize
// =>
// %memref = memref.alloca %arraySize x numFields x elemType
struct LLVMAllocaStruct2MemrefPattern
    : public mlir::OpRewritePattern<mlir::LLVM::AllocaOp> {
  using mlir::OpRewritePattern<mlir::LLVM::AllocaOp>::OpRewritePattern;

  auto matchAndRewrite(mlir::LLVM::AllocaOp allocaOp,
                       mlir::PatternRewriter &rewriter) const
      -> mlir::LogicalResult override {
    auto structType =
        mlir::dyn_cast<mlir::LLVM::LLVMStructType>(allocaOp.getElemType());
    if (!structType) {
      return mlir::failure();
    }

    if (!isStructElemSameType(structType)) {
      return mlir::failure();
    }

    uint32_t numFields = structType.getBody().size();
    auto elemType = structType.getBody()[0];

    mlir::Value arraySizeVal = allocaOp.getArraySize();
    int64_t arraySize = -1;

    if (auto constOp = arraySizeVal.getDefiningOp<mlir::LLVM::ConstantOp>()) {
      if (auto intAttr =
              mlir::dyn_cast<mlir::IntegerAttr>(constOp.getValue())) {
        arraySize = intAttr.getInt();
      }
    }

    mlir::MemRefType memrefType;
    if (arraySize > 0) {
      memrefType = mlir::MemRefType::get(
          {arraySize, static_cast<int64_t>(numFields)}, elemType);
    } else {
      memrefType = mlir::MemRefType::get(
          {mlir::ShapedType::kDynamic, static_cast<int64_t>(numFields)},
          elemType);
    }

    for (mlir::Operation *user :
         llvm::make_early_inc_range(allocaOp->getUsers())) {
      if (auto storeOp = mlir::dyn_cast<mlir::LLVM::StoreOp>(user)) {
        if (storeOp.getAddr() == allocaOp.getResult()) {
          rewriter.eraseOp(storeOp);
        }
      }
    }

    if (arraySize > 0) {
      rewriter.replaceOpWithNewOp<mlir::memref::AllocaOp>(allocaOp, memrefType);
    } else {
      rewriter.replaceOpWithNewOp<mlir::memref::AllocOp>(
          allocaOp, memrefType, mlir::ValueRange{arraySizeVal});
    }

    return mlir::success();
  }
};

// %gep = llvm.getelementptr %base[%i, %field] : (!llvm.ptr) -> !llvm.ptr
// llvm.store %val, %gep
// =>
// memref.store %val, %base[%i, %field] : memref<?xNxelemType>
struct LLVMStoreField2MemrefPattern
    : public mlir::OpRewritePattern<mlir::LLVM::StoreOp> {
  using mlir::OpRewritePattern<mlir::LLVM::StoreOp>::OpRewritePattern;

  auto matchAndRewrite(mlir::LLVM::StoreOp storeOp,
                       mlir::PatternRewriter &rewriter) const
      -> mlir::LogicalResult override {
    auto gepOp = storeOp.getAddr().getDefiningOp<mlir::LLVM::GEPOp>();
    if (!gepOp) {
      return mlir::failure();
    }

    mlir::Value base = gepOp.getBase();
    auto memrefType = mlir::dyn_cast<mlir::MemRefType>(base.getType());
    if (!memrefType) {
      return mlir::failure();
    }

    auto rawIndices = gepOp.getRawConstantIndices();
    if (rawIndices.size() != 2) {
      return mlir::failure();
    }

    mlir::ValueRange dynIndices = gepOp.getDynamicIndices();

    int dynIdx = 0;
    auto resolveIndex = [&](int32_t raw, mlir::PatternRewriter &rw,
                            mlir::Location loc) -> mlir::Value {
      if (raw == mlir::LLVM::GEPOp::kDynamicIndex) {
        return mlir::arith::IndexCastOp::create(rw, loc, rw.getIndexType(),

                                                dynIndices[dynIdx++]);
      }
      return mlir::arith::ConstantIndexOp::create(rw, loc, raw);
    };

    mlir::Location loc = storeOp.getLoc();
    mlir::Value slotIdx = resolveIndex(rawIndices[0], rewriter, loc);
    mlir::Value fieldIdx = resolveIndex(rawIndices[1], rewriter, loc);

    rewriter.replaceOpWithNewOp<mlir::memref::StoreOp>(
        storeOp, storeOp.getValue(), base, mlir::ValueRange{slotIdx, fieldIdx});
    return mlir::success();
  }
};

// %gep = llvm.getelementptr %base[%i, %field] : (!llvm.ptr) -> !llvm.ptr
// %val = llvm.load %gep
// =>
// %val = memref.load %base[%i, %field] : memref<?xNxelemType>
struct LLVMLoadField2MemrefPattern
    : public mlir::OpRewritePattern<mlir::LLVM::LoadOp> {
  using mlir::OpRewritePattern<mlir::LLVM::LoadOp>::OpRewritePattern;

  auto matchAndRewrite(mlir::LLVM::LoadOp loadOp,
                       mlir::PatternRewriter &rewriter) const
      -> mlir::LogicalResult override {
    auto gepOp = loadOp.getAddr().getDefiningOp<mlir::LLVM::GEPOp>();
    if (!gepOp) {
      return mlir::failure();
    }

    mlir::Value base = gepOp.getBase();
    auto memrefType = mlir::dyn_cast<mlir::MemRefType>(base.getType());
    if (!memrefType) {
      return mlir::failure();
    }

    auto rawIndices = gepOp.getRawConstantIndices();
    if (rawIndices.size() != 2) {
      return mlir::failure();
    }

    mlir::ValueRange dynIndices = gepOp.getDynamicIndices();

    int dynIdx = 0;
    auto resolveIndex = [&](int32_t raw, mlir::PatternRewriter &rw,
                            mlir::Location loc) -> mlir::Value {
      if (raw == mlir::LLVM::GEPOp::kDynamicIndex) {
        return mlir::arith::IndexCastOp::create(rw, loc, rw.getIndexType(),
                                                dynIndices[dynIdx++]);
      }
      return mlir::arith::ConstantIndexOp::create(rw, loc, raw);
    };

    mlir::Location loc = loadOp.getLoc();
    mlir::Value slotIdx = resolveIndex(rawIndices[0], rewriter, loc);
    mlir::Value fieldIdx = resolveIndex(rawIndices[1], rewriter, loc);

    rewriter.replaceOpWithNewOp<mlir::memref::LoadOp>(
        loadOp, base, mlir::ValueRange{slotIdx, fieldIdx});
    return mlir::success();
  }
};

// %val = llvm.extractvalue %arg[i] : !llvm.struct<(f32, f32)>
// =>
// %val = memref.load %arg[0, i] : memref<?x2xf32>
struct LLVMExtractValue2MemrefPattern
    : public mlir::OpRewritePattern<mlir::LLVM::ExtractValueOp> {
  using mlir::OpRewritePattern<mlir::LLVM::ExtractValueOp>::OpRewritePattern;

  auto matchAndRewrite(mlir::LLVM::ExtractValueOp extractOp,
                       mlir::PatternRewriter &rewriter) const
      -> mlir::LogicalResult override {
    auto memrefType =
        mlir::dyn_cast<mlir::MemRefType>(extractOp.getContainer().getType());
    if (!memrefType) {
      return mlir::failure();
    }

    if (extractOp.getPosition().size() != 1) {
      return mlir::failure();
    }

    int64_t fieldIdx = extractOp.getPosition()[0];
    mlir::Location loc = extractOp.getLoc();

    auto zero = mlir::arith::ConstantIndexOp::create(rewriter, loc, 0);
    auto field = mlir::arith::ConstantIndexOp::create(rewriter, loc, fieldIdx);

    rewriter.replaceOpWithNewOp<mlir::memref::LoadOp>(
        extractOp, extractOp.getResult().getType(), extractOp.getContainer(),
        mlir::ValueRange{zero, field});
    return mlir::success();
  }
};

struct Struct2MemrefPass
    : public impl::Struct2MemrefPassBase<Struct2MemrefPass> {

  void runOnOperation() override {
    auto op = getOperation();
    mlir::RewritePatternSet patterns(&getContext());

    patterns.add<LLVMStructFuncArgToMemrefPattern>(&getContext());
    patterns.add<LLVMAllocaStruct2MemrefPattern>(&getContext());
    patterns.add<LLVMStoreField2MemrefPattern>(&getContext());
    patterns.add<LLVMLoadField2MemrefPattern>(&getContext());
    patterns.add<LLVMExtractValue2MemrefPattern>(&getContext());

    if (mlir::failed(mlir::applyPatternsGreedily(op, std::move(patterns)))) {
      signalPassFailure();
      return;
    }

    op->walk([](mlir::Operation *op) -> void {
      if (mlir::isOpTriviallyDead(op)) {
        op->erase();
      }
    });
  }
};

auto createStruct2MemrefPass() -> std::unique_ptr<mlir::Pass> {
  return std::make_unique<Struct2MemrefPass>();
}

} // namespace cmlir
