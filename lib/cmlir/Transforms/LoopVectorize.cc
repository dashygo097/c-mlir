#include "cmlir/Transforms/Passes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Pass/Pass.h"

#define GEN_PASS_DEF_LOOPVECTORIZEPASS
#include "cmlir/Transforms/Passes.h.inc"

namespace cmlir {

struct LoopVectorizePass
    : public impl::LoopVectorizePassBase<LoopVectorizePass> {
  void runOnOperation() override {
    getOperation()->walk([&](mlir::scf::ForOp forOp) {
      if (!forOp->hasAttr("vectorize")) {
        return;
      }

      uint32_t width = 4;
      if (auto attr =
              forOp->getAttrOfType<mlir::IntegerAttr>("vectorize_width")) {
        width = attr.getInt();
      }

      mlir::OpBuilder builder(forOp);
      mlir::Location loc = forOp.getLoc();

      mlir::Value oldStep = forOp.getStep();
      mlir::Value widthVal =
          mlir::arith::ConstantIndexOp::create(builder, loc, width);
      mlir::Value newStep =
          mlir::arith::MulIOp::create(builder, loc, oldStep, widthVal);
      forOp.setStep(newStep);

      builder.setInsertionPointToStart(forOp.getBody());
      llvm::DenseMap<mlir::Value, mlir::Value> valueMapping;

      auto getVectorizedOperand = [&](mlir::Value oldOp,
                                      mlir::Type elemType) -> mlir::Value {
        if (valueMapping.count(oldOp)) {
          return valueMapping[oldOp];
        }

        auto vecType = mlir::VectorType::get({width}, elemType);
        mlir::Value broadcast =
            mlir::vector::BroadcastOp::create(builder, loc, vecType, oldOp);
        valueMapping[oldOp] = broadcast;
        return broadcast;
      };

      llvm::SmallVector<mlir::Operation *> opsToErase;

      for (mlir::Operation &op : forOp.getBody()->without_terminator()) {
        builder.setInsertionPoint(&op);

        if (auto loadOp = mlir::dyn_cast<mlir::memref::LoadOp>(&op)) {
          auto memrefType =
              mlir::dyn_cast<mlir::MemRefType>(loadOp.getMemRef().getType());
          auto vecType =
              mlir::VectorType::get({width}, memrefType.getElementType());

          auto vecLoad = mlir::vector::LoadOp::create(
              builder, loc, vecType, loadOp.getMemRef(), loadOp.getIndices());

          valueMapping[loadOp.getResult()] = vecLoad.getResult();
          opsToErase.push_back(&op);
          continue;
        }

        if (auto storeOp = mlir::dyn_cast<mlir::memref::StoreOp>(&op)) {
          mlir::Value oldVal = storeOp.getValue();
          mlir::Value vecVal = getVectorizedOperand(oldVal, oldVal.getType());

          mlir::vector::StoreOp::create(
              builder, loc, vecVal, storeOp.getMemRef(), storeOp.getIndices());

          opsToErase.push_back(&op);
          continue;
        }

        llvm::StringRef dialectName = op.getDialect()->getNamespace();
        if ((dialectName == "arith" || dialectName == "math") &&
            op.getNumResults() == 1) {

          if (mlir::isa<mlir::arith::IndexCastOp>(&op)) {
            continue;
          }

          mlir::Type elemType = op.getResult(0).getType();
          if (elemType.isIntOrFloat() && !elemType.isIndex()) {
            auto vecType = mlir::VectorType::get({width}, elemType);

            llvm::SmallVector<mlir::Value> newOperands;
            for (mlir::Value operand : op.getOperands()) {
              newOperands.push_back(getVectorizedOperand(operand, elemType));
            }

            mlir::OperationState state(loc, op.getName().getStringRef(),
                                       newOperands, {vecType}, op.getAttrs());
            mlir::Operation *newOp = builder.create(state);

            valueMapping[op.getResult(0)] = newOp->getResult(0);
            opsToErase.push_back(&op);
            continue;
          }
        }
      }

      for (auto it = opsToErase.rbegin(); it != opsToErase.rend(); ++it) {
        (*it)->dropAllReferences();
        (*it)->erase();
      }

      forOp->removeAttr("vectorize");
      forOp->removeAttr("vectorize_width");
    });
  }
};

auto createLoopVectorizePass() -> std::unique_ptr<mlir::Pass> {
  return std::make_unique<LoopVectorizePass>();
}

} // namespace cmlir
