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

// if (cond) { alloca = new_val; } (no else)
// %x = load alloca
// =>
// %x = select(cond, new_val, init_val)
struct FlattenSCFIf2ArithSelectPattern
    : public mlir::OpRewritePattern<mlir::scf::IfOp> {
  using mlir::OpRewritePattern<mlir::scf::IfOp>::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::scf::IfOp ifOp,
                  mlir::PatternRewriter &rewriter) const override {

    // Only handle no-else case
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

    for (auto alloca : candidates) {
      mlir::Value allocaVal = alloca.getResult();

      // Find pre-if init store in parent block
      mlir::Value initValue;
      for (mlir::Operation *user : alloca->getUsers()) {
        auto store = mlir::dyn_cast<mlir::memref::StoreOp>(user);
        if (store && store->getBlock() == parentBlock &&
            store->isBeforeInBlock(ifOp.getOperation())) {
          initValue = store.getValue();
          break;
        }
      }
      if (!initValue)
        continue;

      // Find exactly one store inside then block (top-level, not nested)
      mlir::memref::StoreOp innerStore;
      bool valid = true;
      for (mlir::Operation *user : alloca->getUsers()) {
        if (!ifOp->isAncestor(user))
          continue;
        // Any non-store use inside the if is not safe
        auto store = mlir::dyn_cast<mlir::memref::StoreOp>(user);
        if (!store) {
          valid = false;
          break;
        }
        // Must be directly in thenBlock, not in deeper nesting
        if (store->getBlock() != thenBlock) {
          valid = false;
          break;
        }
        // Only allow one inner store
        if (innerStore) {
          valid = false;
          break;
        }
        innerStore = store;
      }
      if (!valid || !innerStore)
        continue;

      mlir::Value storedVal = innerStore.getValue();

      // storedVal must be defined outside thenBlock
      if (auto *defOp = storedVal.getDefiningOp()) {
        if (thenBlock->findAncestorOpInBlock(*defOp) != nullptr)
          continue;
      }

      // heck there are post-if loads to replace
      bool hasPostIfLoad = false;
      for (mlir::Operation *user : allocaVal.getUsers()) {
        if (mlir::isa<mlir::memref::LoadOp>(user) &&
            user->getBlock() == parentBlock && ifOp->isBeforeInBlock(user))
          hasPostIfLoad = true;
      }
      if (!hasPostIfLoad)
        continue;

      // Emit select after the if
      rewriter.setInsertionPointAfter(ifOp);
      mlir::Value selectVal = mlir::arith::SelectOp::create(
                                  rewriter, ifOp.getLoc(), ifOp.getCondition(),
                                  storedVal, initValue)
                                  .getResult();

      // Replace all post-if loads with selectVal
      for (mlir::Operation *user :
           llvm::make_early_inc_range(allocaVal.getUsers())) {
        auto load = mlir::dyn_cast<mlir::memref::LoadOp>(user);
        if (load && load->getBlock() == parentBlock &&
            ifOp->isBeforeInBlock(load.getOperation())) {
          rewriter.replaceOp(load, selectVal);
          changed = true;
        }
      }

      // Remove the now-dead inner store
      rewriter.eraseOp(innerStore);
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

    op->walk([](mlir::Operation *op) {
      if (mlir::isOpTriviallyDead(op)) {
        op->erase();
      }
    });
  }
};

std::unique_ptr<mlir::Pass> createFlattenCondPass() {
  return std::make_unique<FlattenCondPass>();
}

} // namespace cmlir
