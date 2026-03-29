#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Dominance.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#define GEN_PASS_DEF_MEM2REGPASS
#include "cmlir/Transforms/Passes.h.inc"

namespace cmlir {

static auto isScalarMemRef(mlir::Type type) -> bool {
  auto memrefType = mlir::dyn_cast<mlir::MemRefType>(type);
  return memrefType && memrefType.hasRank() && memrefType.getRank() == 0;
}

static auto isInConditionalRegion(mlir::Operation *op) -> bool {
  mlir::Operation *parent = op->getParentOp();
  while (parent) {
    if (mlir::isa<mlir::scf::IfOp, mlir::scf::IndexSwitchOp>(parent)) {
      return true;
}
    if (mlir::isa<mlir::scf::ForOp, mlir::scf::WhileOp, mlir::func::FuncOp>(
            parent)) {
      break;
}
    parent = parent->getParentOp();
  }
  return false;
}

static auto isPromotable(mlir::memref::AllocaOp alloca) -> bool {
  if (!isScalarMemRef(alloca.getType())) {
    return false;
}
  return llvm::all_of(alloca->getUsers(), [](mlir::Operation *user) -> bool {
    if (!mlir::isa<mlir::memref::LoadOp, mlir::memref::StoreOp>(user)) {
      return false;
}
    if (mlir::isa<mlir::memref::StoreOp>(user) && isInConditionalRegion(user)) {
      return false;
}
    return true;
  });
}

static auto getEnclosingBlockInRegion(mlir::Block *block,
                                              mlir::Region *targetRegion) -> mlir::Block * {
  while (block) {
    mlir::Region *r = block->getParent();
    if (!r) {
      return nullptr;
}
    if (r == targetRegion) {
      return block;
}
    mlir::Operation *parentOp = r->getParentOp();
    if (!parentOp) {
      return nullptr;
}
    block = parentOp->getBlock();
  }
  return nullptr;
}

static void replaceNestedLoads(
    mlir::Operation *clonedOp, const llvm::DenseSet<mlir::Value> &promotedSet,
    const llvm::DenseMap<mlir::Value, mlir::Value> &currentValues,
    mlir::PatternRewriter &rewriter) {
  if (clonedOp->getNumRegions() == 0) {
    return;
}
  llvm::SmallVector<mlir::memref::LoadOp> nestedLoads;
  clonedOp->walk([&](mlir::memref::LoadOp nl) -> void {
    if (promotedSet.count(nl.getMemRef())) {
      nestedLoads.push_back(nl);
}
  });
  for (mlir::memref::LoadOp nl : nestedLoads) {
    auto it = currentValues.find(nl.getMemRef());
    if (it != currentValues.end()) {
      rewriter.replaceOp(nl, it->second);
}
  }
}

struct PromoteSCFForAllocaToIterArgPattern
    : public mlir::OpRewritePattern<mlir::scf::ForOp> {
  using mlir::OpRewritePattern<mlir::scf::ForOp>::OpRewritePattern;

  auto
  matchAndRewrite(mlir::scf::ForOp forOp,
                  mlir::PatternRewriter &rewriter) const -> mlir::LogicalResult override {

    auto parentFunc = forOp->getParentOfType<mlir::func::FuncOp>();
    if (!parentFunc) {
      return mlir::failure();
}

    llvm::SmallVector<mlir::memref::AllocaOp> allocasToPromote;
    llvm::SmallVector<mlir::Value> initialValues;

    parentFunc.walk([&](mlir::memref::AllocaOp alloca) -> void {
      if (!isPromotable(alloca)) {
        return;
}
      if (alloca->getBlock() != forOp->getBlock()) {
        return;
}
      if (!alloca->isBeforeInBlock(forOp.getOperation())) {
        return;
}

      bool usedInLoop = false;
      for (mlir::Operation *user : alloca->getUsers()) {
        if (!forOp->isAncestor(user)) {
          continue;
}
        usedInLoop = true;

        if (mlir::isa<mlir::memref::StoreOp>(user)) {
          mlir::Block *enclosing = getEnclosingBlockInRegion(
              user->getBlock(), &forOp.getBodyRegion());
          if (enclosing != forOp.getBody()) {
            return;
}
        }
      }
      if (!usedInLoop) {
        return;
}

      mlir::Value initValue;
      for (mlir::Operation *user : alloca->getUsers()) {
        auto store = mlir::dyn_cast<mlir::memref::StoreOp>(user);
        if (store && store->getBlock() == forOp->getBlock() &&
            store->isBeforeInBlock(forOp.getOperation())) {
          initValue = store.getValue();
          break;
        }
      }
      if (!initValue) {
        return;
}

      allocasToPromote.push_back(alloca);
      initialValues.push_back(initValue);
    });

    if (allocasToPromote.empty()) {
      return mlir::failure();
}

    llvm::SmallVector<mlir::Value> newInitArgs(forOp.getInitArgs().begin(),
                                               forOp.getInitArgs().end());
    newInitArgs.append(initialValues.begin(), initialValues.end());
    size_t numOldIterArgs = forOp.getRegionIterArgs().size();

    auto newFor = mlir::scf::ForOp::create(
        rewriter, forOp.getLoc(), forOp.getLowerBound(), forOp.getUpperBound(),
        forOp.getStep(), newInitArgs);

    mlir::Block *newBody = newFor.getBody();
    if (!newBody->empty() &&
        newBody->back().hasTrait<mlir::OpTrait::IsTerminator>()) {
      rewriter.eraseOp(&newBody->back());
}

    mlir::Block *oldBody = forOp.getBody();

    mlir::IRMapping mapping;
    mapping.map(oldBody->getArgument(0), newBody->getArgument(0));
    for (size_t i = 0; i < numOldIterArgs; ++i) {
      mapping.map(oldBody->getArgument(i + 1), newBody->getArgument(i + 1));
}

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
      if (auto load = mlir::dyn_cast<mlir::memref::LoadOp>(&op)) {
        if (promotedSet.count(load.getMemRef())) {
          mapping.map(load.getResult(), currentValues[load.getMemRef()]);
          continue;
        }
      }
      if (auto store = mlir::dyn_cast<mlir::memref::StoreOp>(&op)) {
        if (promotedSet.count(store.getMemRef())) {
          currentValues[store.getMemRef()] =
              mapping.lookupOrDefault(store.getValue());
          continue;
        }
      }

      mlir::Operation *clonedOp = rewriter.clone(op, mapping);
      replaceNestedLoads(clonedOp, promotedSet, currentValues, rewriter);
    }

    auto oldYield = mlir::cast<mlir::scf::YieldOp>(oldBody->getTerminator());
    llvm::SmallVector<mlir::Value> yieldVals;
    for (mlir::Value v : oldYield.getOperands()) {
      yieldVals.push_back(mapping.lookupOrDefault(v));
}
    for (mlir::memref::AllocaOp alloca : allocasToPromote) {
      yieldVals.push_back(currentValues[alloca.getResult()]);
}
    mlir::scf::YieldOp::create(rewriter, oldYield.getLoc(), yieldVals);

    for (size_t i = 0; i < numOldIterArgs; ++i) {
      rewriter.replaceAllUsesWith(forOp.getResult(i), newFor.getResult(i));
}

    for (size_t i = 0; i < allocasToPromote.size(); ++i) {
      mlir::Value finalVal = newFor.getResults()[numOldIterArgs + i];
      mlir::Value allocaResult = allocasToPromote[i].getResult();
      for (mlir::Operation *user :
           llvm::make_early_inc_range(allocaResult.getUsers())) {
        auto load = mlir::dyn_cast<mlir::memref::LoadOp>(user);
        if (load && load->getBlock() == newFor->getBlock() &&
            newFor->isBeforeInBlock(load.getOperation())) {
          rewriter.replaceOp(load, finalVal);
}
      }
    }

    rewriter.eraseOp(forOp);

    for (mlir::memref::AllocaOp alloca : allocasToPromote) {
      mlir::Value allocaResult = alloca.getResult();
      for (mlir::Operation *user :
           llvm::make_early_inc_range(allocaResult.getUsers())) {
        if (user->getBlock() == alloca->getBlock()) {
          rewriter.eraseOp(user);
}
      }
      if (allocaResult.use_empty()) {
        rewriter.eraseOp(alloca);
}
    }

    return mlir::success();
  }
};

struct PromoteSCFWhileAllocaToIterArgPattern
    : public mlir::OpRewritePattern<mlir::scf::WhileOp> {
  using mlir::OpRewritePattern<mlir::scf::WhileOp>::OpRewritePattern;

  auto
  matchAndRewrite(mlir::scf::WhileOp whileOp,
                  mlir::PatternRewriter &rewriter) const -> mlir::LogicalResult override {

    auto parentFunc = whileOp->getParentOfType<mlir::func::FuncOp>();
    if (!parentFunc) {
      return mlir::failure();
}

    llvm::SmallVector<mlir::memref::AllocaOp> allocasToPromote;
    llvm::SmallVector<mlir::Value> initialValues;

    parentFunc.walk([&](mlir::memref::AllocaOp alloca) -> void {
      if (!isPromotable(alloca)) {
        return;
}
      if (alloca->getBlock() != whileOp->getBlock()) {
        return;
}
      if (!alloca->isBeforeInBlock(whileOp.getOperation())) {
        return;
}

      bool usedInLoop =
          llvm::any_of(alloca->getUsers(), [&](mlir::Operation *u) -> bool {
            return whileOp->isAncestor(u);
          });
      if (!usedInLoop) {
        return;
}

      for (mlir::Operation *user : alloca->getUsers()) {
        if (!mlir::isa<mlir::memref::StoreOp>(user)) {
          continue;
}
        if (!whileOp->isAncestor(user)) {
          continue;
}
        mlir::Block *enclosingBefore =
            getEnclosingBlockInRegion(user->getBlock(), &whileOp.getBefore());
        mlir::Block *enclosingAfter =
            getEnclosingBlockInRegion(user->getBlock(), &whileOp.getAfter());
        if (!enclosingBefore && !enclosingAfter) {
          return;
}
      }

      mlir::Value initValue;
      for (mlir::Operation *user : alloca->getUsers()) {
        auto store = mlir::dyn_cast<mlir::memref::StoreOp>(user);
        if (store && store->getBlock() == whileOp->getBlock() &&
            store->isBeforeInBlock(whileOp.getOperation())) {
          initValue = store.getValue();
          break;
        }
      }
      if (!initValue) {
        return;
}

      allocasToPromote.push_back(alloca);
      initialValues.push_back(initValue);
    });

    if (allocasToPromote.empty()) {
      return mlir::failure();
}

    size_t numOld = whileOp.getOperands().size();
    llvm::SmallVector<mlir::Value> newOperands(whileOp.getOperands().begin(),
                                               whileOp.getOperands().end());
    newOperands.append(initialValues.begin(), initialValues.end());

    llvm::SmallVector<mlir::Type> newResultTypes(
        whileOp.getResultTypes().begin(), whileOp.getResultTypes().end());
    for (auto &v : initialValues) {
      newResultTypes.push_back(v.getType());
}

    auto newWhile = mlir::scf::WhileOp::create(
        rewriter, whileOp.getLoc(), newResultTypes, newOperands,
        [&](mlir::OpBuilder &builder, mlir::Location loc,
            mlir::ValueRange args) -> void {
          mlir::scf::ConditionOp::create(
              builder, loc,
              mlir::arith::ConstantOp::create(builder, loc, builder.getI1Type(),
                                              builder.getBoolAttr(true))
                  .getResult(),
              mlir::ValueRange{});
        },
        [&](mlir::OpBuilder &builder, mlir::Location loc,
            mlir::ValueRange args) -> void {
          mlir::scf::YieldOp::create(builder, loc, mlir::ValueRange{});
        });

    mlir::Block *newBefore = &newWhile.getBefore().front();
    mlir::Block *oldBefore = &whileOp.getBefore().front();
    rewriter.eraseOp(&newBefore->back());

    mlir::IRMapping beforeMapping;
    for (size_t i = 0; i < oldBefore->getNumArguments(); ++i) {
      beforeMapping.map(oldBefore->getArgument(i), newBefore->getArgument(i));
}

    llvm::DenseSet<mlir::Value> promotedSet;
    llvm::DenseMap<mlir::Value, mlir::Value> beforeValues;
    for (size_t i = 0; i < allocasToPromote.size(); ++i) {
      mlir::Value allocaVal = allocasToPromote[i].getResult();
      mlir::BlockArgument arg = newBefore->getArgument(numOld + i);
      promotedSet.insert(allocaVal);
      beforeValues[allocaVal] = arg;
    }

    rewriter.setInsertionPointToEnd(newBefore);
    for (mlir::Operation &op : oldBefore->without_terminator()) {
      if (auto load = mlir::dyn_cast<mlir::memref::LoadOp>(&op)) {
        if (promotedSet.count(load.getMemRef())) {
          beforeMapping.map(load.getResult(), beforeValues[load.getMemRef()]);
          continue;
        }
      }
      if (auto store = mlir::dyn_cast<mlir::memref::StoreOp>(&op)) {
        if (promotedSet.count(store.getMemRef())) {
          beforeValues[store.getMemRef()] =
              beforeMapping.lookupOrDefault(store.getValue());
          continue;
        }
      }
      mlir::Operation *cloned = rewriter.clone(op, beforeMapping);
      replaceNestedLoads(cloned, promotedSet, beforeValues, rewriter);
    }

    auto oldCond =
        mlir::cast<mlir::scf::ConditionOp>(oldBefore->getTerminator());
    llvm::SmallVector<mlir::Value> condArgs;
    for (mlir::Value v : oldCond.getArgs()) {
      condArgs.push_back(beforeMapping.lookupOrDefault(v));
}
    for (auto alloca : allocasToPromote) {
      condArgs.push_back(beforeValues[alloca.getResult()]);
}
    mlir::scf::ConditionOp::create(
        rewriter, oldCond.getLoc(),
        beforeMapping.lookupOrDefault(oldCond.getCondition()), condArgs);

    mlir::Block *newAfter = &newWhile.getAfter().front();
    mlir::Block *oldAfter = &whileOp.getAfter().front();
    rewriter.eraseOp(&newAfter->back());

    mlir::IRMapping afterMapping;
    for (size_t i = 0; i < oldAfter->getNumArguments(); ++i) {
      afterMapping.map(oldAfter->getArgument(i), newAfter->getArgument(i));
}

    llvm::DenseMap<mlir::Value, mlir::Value> afterValues;
    for (size_t i = 0; i < allocasToPromote.size(); ++i) {
      mlir::Value allocaVal = allocasToPromote[i].getResult();
      mlir::BlockArgument arg =
          newAfter->getArgument(oldAfter->getNumArguments() + i);
      afterValues[allocaVal] = arg;
    }

    rewriter.setInsertionPointToEnd(newAfter);
    for (mlir::Operation &op : oldAfter->without_terminator()) {
      if (auto load = mlir::dyn_cast<mlir::memref::LoadOp>(&op)) {
        if (promotedSet.count(load.getMemRef())) {
          afterMapping.map(load.getResult(), afterValues[load.getMemRef()]);
          continue;
        }
      }
      if (auto store = mlir::dyn_cast<mlir::memref::StoreOp>(&op)) {
        if (promotedSet.count(store.getMemRef())) {
          afterValues[store.getMemRef()] =
              afterMapping.lookupOrDefault(store.getValue());
          continue;
        }
      }
      mlir::Operation *cloned = rewriter.clone(op, afterMapping);
      replaceNestedLoads(cloned, promotedSet, afterValues, rewriter);
    }

    auto oldYield = mlir::cast<mlir::scf::YieldOp>(oldAfter->getTerminator());
    llvm::SmallVector<mlir::Value> yieldVals;
    for (mlir::Value v : oldYield.getOperands()) {
      yieldVals.push_back(afterMapping.lookupOrDefault(v));
}
    for (mlir::memref::AllocaOp alloca : allocasToPromote) {
      yieldVals.push_back(afterValues[alloca.getResult()]);
}
    mlir::scf::YieldOp::create(rewriter, oldYield.getLoc(), yieldVals);

    for (size_t i = 0; i < whileOp.getNumResults(); ++i) {
      rewriter.replaceAllUsesWith(whileOp.getResult(i), newWhile.getResult(i));
}

    for (size_t i = 0; i < allocasToPromote.size(); ++i) {
      mlir::Value finalVal = newWhile.getResults()[numOld + i];
      mlir::Value allocaResult = allocasToPromote[i].getResult();
      for (mlir::Operation *user :
           llvm::make_early_inc_range(allocaResult.getUsers())) {
        auto load = mlir::dyn_cast<mlir::memref::LoadOp>(user);
        if (load && load->getBlock() == newWhile->getBlock() &&
            newWhile->isBeforeInBlock(load.getOperation())) {
          rewriter.replaceOp(load, finalVal);
}
      }
    }

    rewriter.eraseOp(whileOp);

    for (mlir::memref::AllocaOp alloca : allocasToPromote) {
      mlir::Value allocaResult = alloca.getResult();
      for (mlir::Operation *user :
           llvm::make_early_inc_range(allocaResult.getUsers())) {
        if (user->getBlock() == alloca->getBlock()) {
          rewriter.eraseOp(user);
}
      }
      if (allocaResult.use_empty()) {
        rewriter.eraseOp(alloca);
}
    }

    return mlir::success();
  }
};

struct ForwardStoreToLoadPattern
    : public mlir::OpRewritePattern<mlir::memref::LoadOp> {
  using mlir::OpRewritePattern<mlir::memref::LoadOp>::OpRewritePattern;

  auto
  matchAndRewrite(mlir::memref::LoadOp loadOp,
                  mlir::PatternRewriter &rewriter) const -> mlir::LogicalResult override {

    auto alloca = loadOp.getMemRef().getDefiningOp<mlir::memref::AllocaOp>();
    if (!alloca || !isScalarMemRef(alloca.getType())) {
      return mlir::failure();
}

    llvm::SmallVector<mlir::memref::StoreOp> stores;
    for (mlir::Operation *user : alloca->getUsers()) {
      if (auto store = mlir::dyn_cast<mlir::memref::StoreOp>(user)) {
        stores.push_back(store);
      } else if (!mlir::isa<mlir::memref::LoadOp>(user)) {
        return mlir::failure();
}
    }

    if (stores.size() != 1) {
      return mlir::failure();
}

    mlir::memref::StoreOp store = stores[0];
    mlir::DominanceInfo domInfo(alloca->getParentOfType<mlir::func::FuncOp>());

    if (!domInfo.dominates(store.getOperation(), loadOp.getOperation())) {
      return mlir::failure();
}

    rewriter.replaceOp(loadOp, store.getValue());
    return mlir::success();
  }
};

struct Mem2RegPass : public impl::Mem2RegPassBase<Mem2RegPass> {
  void runOnOperation() override {
    auto op = getOperation();
    mlir::RewritePatternSet patterns(op->getContext());

    patterns.add<PromoteSCFForAllocaToIterArgPattern>(op->getContext());
    patterns.add<PromoteSCFWhileAllocaToIterArgPattern>(op->getContext());
    patterns.add<ForwardStoreToLoadPattern>(op->getContext());

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

auto createMem2RegPass() -> std::unique_ptr<mlir::Pass> {
  return std::make_unique<Mem2RegPass>();
}

} // namespace cmlir
