#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Dominance.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#define GEN_PASS_DEF_FLATTENCONDPASS
#include "cmlir/Transforms/Passes.h.inc"

namespace cmlir {

static bool isScalarMemRef(mlir::Type type) {
  auto memrefType = mlir::dyn_cast<mlir::MemRefType>(type);
  return memrefType && memrefType.hasRank() && memrefType.getRank() == 0;
}

static bool isPromotable(mlir::memref::AllocaOp alloca) {
  if (!isScalarMemRef(alloca.getType()))
    return false;
  for (mlir::Operation *user : alloca->getUsers())
    if (!mlir::isa<mlir::memref::LoadOp, mlir::memref::StoreOp>(user))
      return false;
  return true;
}

// %alloca = memref.alloca() : memref<T>
// memref.store %init, %alloca[]
// scf.if %cond {
//   memref.store %new_val, %alloca[]
// }
// %x = memref.load %alloca[]
// =>
// %sel = arith.select %cond, %new_val, %init : T
// memref.store %sel, %alloca[]
struct FlattenSCFIf2ArithSelectPattern
    : public mlir::OpRewritePattern<mlir::scf::IfOp> {
  using mlir::OpRewritePattern<mlir::scf::IfOp>::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::scf::IfOp ifOp,
                  mlir::PatternRewriter &rewriter) const override {

    // FIXME: Only handle the no-else case.
    if (ifOp.elseBlock())
      return mlir::failure();

    auto parentFunc = ifOp->getParentOfType<mlir::func::FuncOp>();
    if (!parentFunc)
      return mlir::failure();

    mlir::Block *thenBlock = ifOp.thenBlock();
    mlir::Block *parentBlock = ifOp->getBlock();
    bool changed = false;

    // Collect candidate allocas
    llvm::SmallVector<mlir::memref::AllocaOp> candidates;
    parentFunc.walk([&](mlir::memref::AllocaOp alloca) {
      if (!isPromotable(alloca))
        return;
      if (alloca->getBlock() != parentBlock)
        return;
      if (!alloca->isBeforeInBlock(ifOp.getOperation()))
        return;
      candidates.push_back(alloca);
    });

    // insertAfter tracks where to place the next select + update store.
    mlir::Operation *insertAfter = ifOp.getOperation();

    for (mlir::memref::AllocaOp alloca : candidates) {
      mlir::Value allocaVal = alloca.getResult();

      // Find the latest pre-if initialising store
      mlir::memref::StoreOp initStore;
      for (mlir::Operation *user : alloca->getUsers()) {
        auto store = mlir::dyn_cast<mlir::memref::StoreOp>(user);
        if (!store)
          continue;
        if (store->getBlock() != parentBlock)
          continue;
        if (!store->isBeforeInBlock(ifOp.getOperation()))
          continue;
        if (!initStore || initStore->isBeforeInBlock(store.getOperation()))
          initStore = store;
      }
      if (!initStore)
        continue;
      mlir::Value initValue = initStore.getValue();

      // Require exactly one top-level store inside thenBlock
      mlir::memref::StoreOp innerStore;
      bool valid = true;
      for (mlir::Operation *user : alloca->getUsers()) {
        if (!ifOp->isAncestor(user))
          continue;
        auto store = mlir::dyn_cast<mlir::memref::StoreOp>(user);
        if (!store) {
          valid = false;
          break;
        }
        if (store->getBlock() != thenBlock) {
          valid = false;
          break;
        }
        if (innerStore) {
          valid = false;
          break;
        }
        innerStore = store;
      }
      if (!valid || !innerStore)
        continue;

      mlir::Value storedVal = innerStore.getValue();

      if (auto *defOp = storedVal.getDefiningOp())
        if (thenBlock->findAncestorOpInBlock(*defOp) != nullptr)
          continue;

      // Collect post-if loads that will be replaced
      llvm::SmallVector<mlir::memref::LoadOp> postIfLoads;
      for (mlir::Operation *user : allocaVal.getUsers()) {
        auto load = mlir::dyn_cast<mlir::memref::LoadOp>(user);
        if (load && load->getBlock() == parentBlock &&
            ifOp->isBeforeInBlock(load.getOperation()))
          postIfLoads.push_back(load);
      }
      if (postIfLoads.empty())
        continue;

      // Emit arith.select
      rewriter.setInsertionPointAfter(insertAfter);
      mlir::Value selectVal = mlir::arith::SelectOp::create(
                                  rewriter, ifOp.getLoc(), ifOp.getCondition(),
                                  storedVal, initValue)
                                  .getResult();

      mlir::memref::StoreOp updateStore = mlir::memref::StoreOp::create(
          rewriter, ifOp.getLoc(), selectVal, allocaVal, mlir::ValueRange{});

      insertAfter = updateStore.getOperation();

      for (mlir::memref::LoadOp load : postIfLoads) {
        rewriter.replaceOp(load, selectVal);
        changed = true;
      }

      rewriter.eraseOp(innerStore);
    }

    // Erase the scf.if if its body is now a bare scf.yield
    if (changed) {
      bool thenIsEmpty = (thenBlock->getOperations().size() == 1 &&
                          mlir::isa<mlir::scf::YieldOp>(thenBlock->front()));
      if (thenIsEmpty)
        rewriter.eraseOp(ifOp);
    }

    return changed ? mlir::success() : mlir::failure();
  }
};

struct FlattenCondPass : public impl::FlattenCondPassBase<FlattenCondPass> {
  void runOnOperation() override {
    auto op = getOperation();
    mlir::RewritePatternSet patterns(op->getContext());
    patterns.add<FlattenSCFIf2ArithSelectPattern>(op->getContext());

    if (mlir::failed(mlir::applyPatternsGreedily(op, std::move(patterns)))) {
      signalPassFailure();
      return;
    }

    op->walk([](mlir::Operation *inner) {
      if (mlir::isOpTriviallyDead(inner))
        inner->erase();
    });
  }
};

std::unique_ptr<mlir::Pass> createFlattenCondPass() {
  return std::make_unique<FlattenCondPass>();
}

} // namespace cmlir
