#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Dominance.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#define GEN_PASS_DEF_MEM2REGPASS
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

struct PromoteAllocaToIterArgPattern
    : public mlir::OpRewritePattern<mlir::scf::ForOp> {
  using mlir::OpRewritePattern<mlir::scf::ForOp>::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::scf::ForOp forOp,
                  mlir::PatternRewriter &rewriter) const override {

    auto parentFunc = forOp->getParentOfType<mlir::func::FuncOp>();
    if (!parentFunc)
      return mlir::failure();

    llvm::SmallVector<mlir::memref::AllocaOp> allocasToPromote;
    llvm::SmallVector<mlir::Value> initialValues;

    parentFunc.walk([&](mlir::memref::AllocaOp alloca) {
      if (!isPromotable(alloca))
        return;
      if (alloca->getBlock() != forOp->getBlock())
        return;
      if (!alloca->isBeforeInBlock(forOp.getOperation()))
        return;

      bool usedInLoop = false;
      for (mlir::Operation *user : alloca->getUsers()) {
        if (!forOp->isAncestor(user))
          continue;
        usedInLoop = true;

        if (mlir::isa<mlir::memref::StoreOp>(user) &&
            user->getBlock() != forOp.getBody())
          return;
      }
      if (!usedInLoop)
        return;

      // Must have a pre-loop initialising store
      mlir::Value initValue;
      for (mlir::Operation *user : alloca->getUsers()) {
        auto store = mlir::dyn_cast<mlir::memref::StoreOp>(user);
        if (store && store->getBlock() == forOp->getBlock() &&
            store->isBeforeInBlock(forOp.getOperation())) {
          initValue = store.getValue();
          break;
        }
      }
      if (!initValue)
        return;

      allocasToPromote.push_back(alloca);
      initialValues.push_back(initValue);
    });

    if (allocasToPromote.empty())
      return mlir::failure();

    llvm::SmallVector<mlir::Value> newInitArgs(forOp.getInitArgs().begin(),
                                               forOp.getInitArgs().end());
    newInitArgs.append(initialValues.begin(), initialValues.end());
    size_t numOldIterArgs = forOp.getRegionIterArgs().size();

    auto newFor = mlir::scf::ForOp::create(
        rewriter, forOp.getLoc(), forOp.getLowerBound(), forOp.getUpperBound(),
        forOp.getStep(), newInitArgs);

    mlir::Block *newBody = newFor.getBody();
    if (!newBody->empty() &&
        newBody->back().hasTrait<mlir::OpTrait::IsTerminator>())
      rewriter.eraseOp(&newBody->back());

    mlir::Block *oldBody = forOp.getBody();

    mlir::IRMapping mapping;
    mapping.map(oldBody->getArgument(0), newBody->getArgument(0));
    for (size_t i = 0; i < numOldIterArgs; ++i)
      mapping.map(oldBody->getArgument(i + 1), newBody->getArgument(i + 1));

    llvm::DenseSet<mlir::Value> promotedSet;
    llvm::DenseMap<mlir::Value, mlir::Value> currentValues;
    for (size_t i = 0; i < allocasToPromote.size(); ++i) {
      mlir::Value allocaVal = allocasToPromote[i].getResult();
      mlir::BlockArgument iterArg =
          newBody->getArgument(numOldIterArgs + i + 1);
      promotedSet.insert(allocaVal);
      currentValues[allocaVal] = iterArg;
    }

    rewriter.setInsertionPointToEnd(newBody);

    for (mlir::Operation &op : oldBody->without_terminator()) {
      // Direct (top-level) promoted load → SSA value
      if (auto load = mlir::dyn_cast<mlir::memref::LoadOp>(&op)) {
        if (promotedSet.count(load.getMemRef())) {
          mapping.map(load.getResult(), currentValues[load.getMemRef()]);
          continue;
        }
      }
      // Direct (top-level) promoted store → update current value
      if (auto store = mlir::dyn_cast<mlir::memref::StoreOp>(&op)) {
        if (promotedSet.count(store.getMemRef())) {
          currentValues[store.getMemRef()] =
              mapping.lookupOrDefault(store.getValue());
          continue;
        }
      }

      // Clone op
      mlir::Operation *clonedOp = rewriter.clone(op, mapping);

      if (clonedOp->getNumRegions() > 0) {
        llvm::SmallVector<mlir::memref::LoadOp> nestedLoads;
        clonedOp->walk([&](mlir::memref::LoadOp nestedLoad) {
          if (promotedSet.count(nestedLoad.getMemRef()))
            nestedLoads.push_back(nestedLoad);
        });
        for (mlir::memref::LoadOp nestedLoad : nestedLoads) {
          auto it = currentValues.find(nestedLoad.getMemRef());
          if (it != currentValues.end())
            rewriter.replaceOp(nestedLoad, it->second);
        }
      }
    }

    auto oldYield = mlir::cast<mlir::scf::YieldOp>(oldBody->getTerminator());
    llvm::SmallVector<mlir::Value> yieldVals;
    for (mlir::Value v : oldYield.getOperands())
      yieldVals.push_back(mapping.lookupOrDefault(v));
    for (mlir::memref::AllocaOp alloca : allocasToPromote)
      yieldVals.push_back(currentValues[alloca.getResult()]);
    mlir::scf::YieldOp::create(rewriter, oldYield.getLoc(), yieldVals);

    for (size_t i = 0; i < numOldIterArgs; ++i)
      rewriter.replaceAllUsesWith(forOp.getResult(i), newFor.getResult(i));

    for (size_t i = 0; i < allocasToPromote.size(); ++i) {
      mlir::Value finalVal = newFor.getResults()[numOldIterArgs + i];
      mlir::Value allocaResult = allocasToPromote[i].getResult();
      for (mlir::Operation *user :
           llvm::make_early_inc_range(allocaResult.getUsers())) {
        auto load = mlir::dyn_cast<mlir::memref::LoadOp>(user);
        if (load && load->getBlock() == newFor->getBlock() &&
            newFor->isBeforeInBlock(load.getOperation()))
          rewriter.replaceOp(load, finalVal);
      }
    }

    rewriter.eraseOp(forOp);

    for (mlir::memref::AllocaOp alloca : allocasToPromote) {
      for (mlir::Operation *user :
           llvm::make_early_inc_range(alloca->getUsers()))
        rewriter.eraseOp(user);
      rewriter.eraseOp(alloca);
    }

    return mlir::success();
  }
};

// %alloca = memref.alloca() : memref<T>
// memref.store %val, %alloca[]
// ...no other stores...
// =>
// %x = memref.load %alloca[]   →  replace %x with %val (across any nesting
// depth)
struct ForwardStoreToLoadPattern
    : public mlir::OpRewritePattern<mlir::memref::LoadOp> {
  using mlir::OpRewritePattern<mlir::memref::LoadOp>::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::memref::LoadOp loadOp,
                  mlir::PatternRewriter &rewriter) const override {

    auto alloca = loadOp.getMemRef().getDefiningOp<mlir::memref::AllocaOp>();
    if (!alloca || !isScalarMemRef(alloca.getType()))
      return mlir::failure();

    llvm::SmallVector<mlir::memref::StoreOp> stores;
    for (mlir::Operation *user : alloca->getUsers()) {
      if (auto store = mlir::dyn_cast<mlir::memref::StoreOp>(user))
        stores.push_back(store);
      else if (!mlir::isa<mlir::memref::LoadOp>(user))
        return mlir::failure();
    }

    if (stores.size() != 1)
      return mlir::failure();

    mlir::memref::StoreOp store = stores[0];

    mlir::DominanceInfo domInfo(alloca->getParentOfType<mlir::func::FuncOp>());

    if (!domInfo.dominates(store.getOperation(), loadOp.getOperation()))
      return mlir::failure();

    rewriter.replaceOp(loadOp, store.getValue());
    return mlir::success();
  }
};

struct Mem2RegPass : public impl::Mem2RegPassBase<Mem2RegPass> {
  void runOnOperation() override {
    auto op = getOperation();
    mlir::RewritePatternSet patterns(op->getContext());

    patterns.add<PromoteAllocaToIterArgPattern>(op->getContext());
    patterns.add<ForwardStoreToLoadPattern>(op->getContext());

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

std::unique_ptr<mlir::Pass> createMem2RegPass() {
  return std::make_unique<Mem2RegPass>();
}

} // namespace cmlir
