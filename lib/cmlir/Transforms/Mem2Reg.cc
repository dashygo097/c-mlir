#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Dominance.h"
#include "mlir/Pass/Pass.h"

#define GEN_PASS_DEF_MEM2REGPASS
#include "cmlir/Transforms/Passes.h.inc"

namespace cmlir {

// Check if memref is a scalar (rank-0)
bool isScalarMemRef(mlir::Type type) {
  auto memrefType = mlir::dyn_cast<mlir::MemRefType>(type);
  return memrefType && memrefType.hasRank() && memrefType.getRank() == 0;
}

// Check if all uses are load/store
bool isPromotable(mlir::memref::AllocaOp alloca) {
  if (!isScalarMemRef(alloca.getType()))
    return false;

  for (mlir::Operation *user : alloca->getUsers()) {
    if (!mlir::isa<mlir::memref::LoadOp, mlir::memref::StoreOp>(user))
      return false;
  }
  return true;
}

// Recursively replace loads/stores in a region
void replaceLoadsStores(mlir::Region &region,
                        const llvm::DenseSet<mlir::Value> &promotedAllocas,
                        llvm::DenseMap<mlir::Value, mlir::Value> &currentValues,
                        mlir::IRMapping &mapping) {
  for (mlir::Block &block : region) {
    for (mlir::Operation &op : llvm::make_early_inc_range(block)) {
      // Handle loads
      if (auto load = mlir::dyn_cast<mlir::memref::LoadOp>(&op)) {
        if (promotedAllocas.count(load.getMemRef())) {
          mlir::Value replacement = currentValues[load.getMemRef()];
          mapping.map(load.getResult(), replacement);
          load.getResult().replaceAllUsesWith(replacement);
          load.erase();
          continue;
        }
      }

      // Handle stores
      if (auto store = mlir::dyn_cast<mlir::memref::StoreOp>(&op)) {
        if (promotedAllocas.count(store.getMemRef())) {
          mlir::Value newValue = mapping.lookupOrDefault(store.getValue());
          currentValues[store.getMemRef()] = newValue;
          store.erase();
          continue;
        }
      }

      // Recursively handle nested regions
      for (mlir::Region &nestedRegion : op.getRegions()) {
        replaceLoadsStores(nestedRegion, promotedAllocas, currentValues,
                           mapping);
      }
    }
  }
}

struct Mem2RegPass
    : public mlir::PassWrapper<Mem2RegPass,
                               mlir::OperationPass<mlir::func::FuncOp>> {

  // Process a single loop, returns true if modified
  bool processLoop(mlir::scf::ForOp forOp) {
    llvm::SmallVector<mlir::memref::AllocaOp> allocasToPromote;
    llvm::SmallVector<mlir::Value> initialValues;

    // Collect all scalar allocas before this loop
    llvm::SmallVector<mlir::memref::AllocaOp> candidateAllocas;
    forOp->getParentOfType<mlir::func::FuncOp>().walk(
        [&](mlir::memref::AllocaOp alloca) {
          if (isPromotable(alloca) && alloca->getBlock() == forOp->getBlock() &&
              alloca->isBeforeInBlock(forOp.getOperation()))
            candidateAllocas.push_back(alloca);
        });

    for (mlir::memref::AllocaOp alloca : candidateAllocas) {
      // Check if used anywhere in this loop (including nested regions)
      bool usedInLoop = false;
      for (mlir::Operation *user : alloca->getUsers()) {
        if (forOp->isAncestor(user)) {
          usedInLoop = true;
          break;
        }
      }

      if (!usedInLoop)
        continue;

      // Find initial store before the loop
      mlir::Value initValue = nullptr;
      for (mlir::Operation *user : alloca->getUsers()) {
        if (auto store = mlir::dyn_cast<mlir::memref::StoreOp>(user)) {
          if (store->getBlock() == forOp->getBlock() &&
              store->isBeforeInBlock(forOp.getOperation())) {
            initValue = store.getValue();
            break;
          }
        }
      }

      if (!initValue)
        continue;

      allocasToPromote.push_back(alloca);
      initialValues.push_back(initValue);
    }

    if (allocasToPromote.empty())
      return false;

    // Create new for loop with additional iter args
    mlir::OpBuilder builder(forOp);
    llvm::SmallVector<mlir::Value> newInitArgs(forOp.getInitArgs().begin(),
                                               forOp.getInitArgs().end());
    newInitArgs.append(initialValues.begin(), initialValues.end());

    auto newForOp = mlir::scf::ForOp::create(
        builder, forOp.getLoc(), forOp.getLowerBound(), forOp.getUpperBound(),
        forOp.getStep(), newInitArgs);

    // Clone the loop body
    mlir::IRMapping mapping;
    mlir::Block *oldBody = forOp.getBody();
    mlir::Block *newBody = newForOp.getBody();

    // Map induction variable
    mapping.map(oldBody->getArgument(0), newBody->getArgument(0));

    // Map old iter args
    size_t numOldIterArgs = forOp.getRegionIterArgs().size();
    for (size_t i = 0; i < numOldIterArgs; ++i)
      mapping.map(oldBody->getArgument(i + 1), newBody->getArgument(i + 1));

    // Setup current values for promoted allocas
    llvm::DenseSet<mlir::Value> promotedAllocaSet;
    llvm::DenseMap<mlir::Value, mlir::Value> currentValues;
    for (size_t i = 0; i < allocasToPromote.size(); ++i) {
      mlir::Value allocaVal = allocasToPromote[i].getResult();
      mlir::BlockArgument iterArg =
          newBody->getArgument(numOldIterArgs + i + 1);
      promotedAllocaSet.insert(allocaVal);
      currentValues[allocaVal] = iterArg;
    }

    // Clone the body without the terminator
    builder.setInsertionPointToStart(newBody);
    for (mlir::Operation &op : oldBody->without_terminator()) {
      builder.clone(op, mapping);
    }

    // Now recursively replace loads/stores in the cloned body
    replaceLoadsStores(newForOp.getRegion(), promotedAllocaSet, currentValues,
                       mapping);

    // Create new yield
    auto oldYield = mlir::cast<mlir::scf::YieldOp>(oldBody->getTerminator());
    llvm::SmallVector<mlir::Value> yieldVals;
    for (mlir::Value val : oldYield.getOperands())
      yieldVals.push_back(mapping.lookupOrDefault(val));
    for (mlir::Value allocaVal :
         llvm::map_range(allocasToPromote, [](mlir::memref::AllocaOp a) {
           return a.getResult();
         }))
      yieldVals.push_back(currentValues[allocaVal]);

    builder.setInsertionPointToEnd(newBody);
    mlir::scf::YieldOp::create(builder, newForOp.getLoc(), yieldVals);

    // Replace old loop results
    for (size_t i = 0; i < forOp.getNumResults(); ++i)
      forOp.getResult(i).replaceAllUsesWith(newForOp.getResult(i));

    // Replace post-loop loads
    for (size_t i = 0; i < allocasToPromote.size(); ++i) {
      mlir::Value finalValue = newForOp.getResults()[numOldIterArgs + i];
      mlir::Value allocaResult = allocasToPromote[i].getResult();

      for (mlir::Operation *user :
           llvm::make_early_inc_range(allocaResult.getUsers())) {
        if (auto load = mlir::dyn_cast<mlir::memref::LoadOp>(user)) {
          if (load->getBlock() == newForOp->getBlock() &&
              newForOp->isBeforeInBlock(load.getOperation())) {
            load.replaceAllUsesWith(finalValue);
            load.erase();
          }
        }
      }
    }

    // Erase old loop
    forOp.erase();

    // Clean up allocas
    for (mlir::memref::AllocaOp alloca : allocasToPromote) {
      for (mlir::Operation *user :
           llvm::make_early_inc_range(alloca->getUsers()))
        user->erase();
      if (alloca.use_empty())
        alloca.erase();
    }

    return true;
  }

  void runOnOperation() override {
    mlir::func::FuncOp func = getOperation();

    bool changed = true;
    while (changed) {
      changed = false;

      llvm::SmallVector<mlir::scf::ForOp> forLoops;
      func.walk<mlir::WalkOrder::PostOrder>(
          [&](mlir::scf::ForOp forOp) { forLoops.push_back(forOp); });

      for (mlir::scf::ForOp forOp : forLoops) {
        if (!forOp->getBlock())
          continue;

        if (processLoop(forOp)) {
          changed = true;
          break;
        }
      }
    }
  }
};

std::unique_ptr<mlir::OperationPass<mlir::func::FuncOp>> createMem2RegPass() {
  return std::make_unique<Mem2RegPass>();
}

} // namespace cmlir
