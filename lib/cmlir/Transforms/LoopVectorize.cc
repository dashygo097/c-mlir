#include "cmlir/Transforms/Passes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/DenseMap.h"

#include <optional>

#define GEN_PASS_DEF_LOOPVECTORIZEPASS
#include "cmlir/Transforms/Passes.h.inc"

namespace cmlir {

struct VectorBuildState {
  mlir::OpBuilder &builder;
  mlir::Location loc;
  mlir::scf::ForOp oldFor;
  mlir::Value newIv;
  uint32_t width;
  llvm::DenseMap<mlir::Value, mlir::Value> map;

  VectorBuildState(mlir::OpBuilder &builder, mlir::Location loc,
                   mlir::scf::ForOp oldFor, mlir::Value newIv, uint32_t width)
      : builder(builder), loc(loc), oldFor(oldFor), newIv(newIv), width(width),
        map() {}
};

static void clearLoopBody(mlir::scf::ForOp forOp) {
  mlir::Block *body = forOp.getBody();

  while (!body->empty()) {
    body->back().erase();
  }
}

static void transferUnrollAttrsToVectorLoop(mlir::scf::ForOp oldFor,
                                            mlir::scf::ForOp vectorFor) {
  if (oldFor->hasAttr("nounroll")) {
    return;
  }

  if (auto attr = oldFor->getAttr("unroll_count")) {
    vectorFor->setAttr("unroll_count", attr);
    return;
  }

  if (auto attr = oldFor->getAttr("unroll")) {
    vectorFor->setAttr("unroll", attr);
  }
}

static auto getConstantIndexValue(mlir::Value value) -> std::optional<int64_t> {
  auto op = value.getDefiningOp<mlir::arith::ConstantIndexOp>();
  if (!op) {
    return std::nullopt;
  }

  return op.value();
}

static auto isOne(mlir::Value value) -> bool {
  auto v = getConstantIndexValue(value);
  return v && *v == 1;
}

static auto stripIndexCastRoundTrip(mlir::Value value) -> mlir::Value {
  auto outer = value.getDefiningOp<mlir::arith::IndexCastOp>();
  if (!outer) {
    return value;
  }

  mlir::Value innerValue = outer.getIn();
  auto inner = innerValue.getDefiningOp<mlir::arith::IndexCastOp>();
  if (!inner) {
    return value;
  }

  if (value.getType().isIndex() && inner.getIn().getType().isIndex()) {
    return inner.getIn();
  }

  return value;
}

static auto createZeroLike(mlir::OpBuilder &builder, mlir::Location loc,
                           mlir::Type type) -> mlir::Value {
  if (type.isF32() || type.isF64()) {
    return mlir::arith::ConstantOp::create(builder, loc, type,
                                           builder.getZeroAttr(type));
  }

  return {};
}

static auto getYieldOp(mlir::scf::ForOp forOp) -> mlir::scf::YieldOp {
  mlir::Operation *term = forOp.getBody()->getTerminator();
  if (!term) {
    return {};
  }

  return mlir::dyn_cast<mlir::scf::YieldOp>(term);
}

static auto broadcastScalar(VectorBuildState &state, mlir::Value value)
    -> mlir::Value {
  if (!value.getType().isIntOrFloat()) {
    return {};
  }

  auto vecType = mlir::VectorType::get({state.width}, value.getType());

  return mlir::vector::BroadcastOp::create(state.builder, state.loc, vecType,
                                           value);
}

static auto isSupportedArithOp(mlir::Operation *op) -> bool {
  return mlir::isa<
      mlir::arith::AddFOp, mlir::arith::SubFOp, mlir::arith::MulFOp,
      mlir::arith::DivFOp, mlir::arith::AddIOp, mlir::arith::SubIOp,
      mlir::arith::MulIOp, mlir::arith::DivSIOp, mlir::arith::DivUIOp>(op);
}

static auto vectorizeValue(VectorBuildState &state, mlir::Value value)
    -> mlir::Value {
  if (state.map.count(value)) {
    return state.map[value];
  }

  if (value == state.oldFor.getInductionVar()) {
    return broadcastScalar(state, state.newIv);
  }

  mlir::Operation *def = value.getDefiningOp();
  if (!def || def->getBlock() != state.oldFor.getBody()) {
    return broadcastScalar(state, value);
  }

  if (auto indexCast = mlir::dyn_cast<mlir::arith::IndexCastOp>(def)) {
    mlir::Value input = vectorizeValue(state, indexCast.getIn());
    if (!input) {
      return {};
    }

    auto shapedType = mlir::dyn_cast<mlir::ShapedType>(input.getType());
    if (!shapedType) {
      return {};
    }

    auto vecType = mlir::VectorType::get({state.width}, indexCast.getType());

    auto newCast = mlir::arith::IndexCastOp::create(state.builder, state.loc,
                                                    vecType, input);

    state.map[value] = newCast.getResult();
    return newCast.getResult();
  }

  if (auto loadOp = mlir::dyn_cast<mlir::memref::LoadOp>(def)) {
    if (loadOp.getIndices().size() != 1) {
      return {};
    }

    mlir::Value index = stripIndexCastRoundTrip(loadOp.getIndices()[0]);
    if (index != state.oldFor.getInductionVar()) {
      return {};
    }

    auto memrefType =
        mlir::dyn_cast<mlir::MemRefType>(loadOp.getMemRef().getType());
    if (!memrefType) {
      return {};
    }

    auto elemType = memrefType.getElementType();
    if (!elemType.isIntOrFloat()) {
      return {};
    }

    auto vecType = mlir::VectorType::get({state.width}, elemType);

    auto vecLoad = mlir::vector::LoadOp::create(state.builder, state.loc,
                                                vecType, loadOp.getMemRef(),
                                                mlir::ValueRange{state.newIv});

    state.map[value] = vecLoad.getResult();
    return vecLoad.getResult();
  }

  if (isSupportedArithOp(def)) {
    llvm::SmallVector<mlir::Value, 4> newOperands;

    for (mlir::Value operand : def->getOperands()) {
      mlir::Value newOperand = vectorizeValue(state, operand);
      if (!newOperand) {
        return {};
      }

      newOperands.push_back(newOperand);
    }

    auto shapedType =
        mlir::dyn_cast<mlir::ShapedType>(newOperands[0].getType());
    if (!shapedType) {
      return {};
    }

    auto vecType =
        mlir::VectorType::get({state.width}, shapedType.getElementType());

    mlir::OperationState opState(state.loc, def->getName().getStringRef());
    opState.addOperands(newOperands);
    opState.addTypes(vecType);
    opState.addAttributes(def->getAttrs());

    mlir::Operation *newOp = state.builder.create(opState);
    state.map[value] = newOp->getResult(0);
    return newOp->getResult(0);
  }

  return {};
}

static auto isAllowedMapBodyOp(mlir::Operation *op) -> bool {
  if (mlir::isa<mlir::memref::LoadOp, mlir::memref::StoreOp,
                mlir::arith::IndexCastOp>(op)) {
    return true;
  }

  if (isSupportedArithOp(op)) {
    return true;
  }

  return false;
}

static auto isStoreToLoopIv(mlir::memref::StoreOp storeOp,
                            mlir::scf::ForOp forOp) -> bool {
  if (storeOp.getIndices().size() != 1) {
    return false;
  }

  mlir::Value index = stripIndexCastRoundTrip(storeOp.getIndices()[0]);
  return index == forOp.getInductionVar();
}

static auto hasVectorizableMapBody(mlir::scf::ForOp forOp) -> bool {
  if (forOp.getNumRegionIterArgs() != 0 || forOp.getNumResults() != 0) {
    return false;
  }

  auto yieldOp = getYieldOp(forOp);
  if (!yieldOp || yieldOp.getNumOperands() != 0) {
    return false;
  }

  bool hasStore = false;

  for (mlir::Operation &op : forOp.getBody()->without_terminator()) {
    if (!isAllowedMapBodyOp(&op)) {
      return false;
    }

    if (auto storeOp = mlir::dyn_cast<mlir::memref::StoreOp>(&op)) {
      if (!isStoreToLoopIv(storeOp, forOp)) {
        return false;
      }

      hasStore = true;
    }
  }

  return hasStore;
}

static auto vectorizeMapBody(VectorBuildState &state) -> mlir::LogicalResult {
  bool emittedStore = false;

  for (mlir::Operation &op : state.oldFor.getBody()->without_terminator()) {
    if (auto storeOp = mlir::dyn_cast<mlir::memref::StoreOp>(&op)) {
      if (!isStoreToLoopIv(storeOp, state.oldFor)) {
        return mlir::failure();
      }

      mlir::Value vectorValue = vectorizeValue(state, storeOp.getValue());
      if (!vectorValue) {
        return mlir::failure();
      }

      auto vectorType = mlir::dyn_cast<mlir::VectorType>(vectorValue.getType());
      if (!vectorType) {
        return mlir::failure();
      }

      mlir::vector::StoreOp::create(state.builder, state.loc, vectorValue,
                                    storeOp.getMemRef(),
                                    mlir::ValueRange{state.newIv});

      emittedStore = true;
      continue;
    }

    if (!isAllowedMapBodyOp(&op)) {
      return mlir::failure();
    }
  }

  return emittedStore ? mlir::success() : mlir::failure();
}

static auto
buildScalarMapRemainderLoop(mlir::OpBuilder &builder, mlir::Location loc,
                            mlir::scf::ForOp oldFor, mlir::Value lowerBound)
    -> mlir::scf::ForOp {
  auto remainderFor = mlir::scf::ForOp::create(
      builder, loc, lowerBound, oldFor.getUpperBound(), oldFor.getStep());

  clearLoopBody(remainderFor);
  builder.setInsertionPointToEnd(remainderFor.getBody());

  mlir::IRMapping mapper;
  mapper.map(oldFor.getInductionVar(), remainderFor.getInductionVar());

  for (mlir::Operation &op : oldFor.getBody()->without_terminator()) {
    builder.clone(op, mapper);
  }

  mlir::scf::YieldOp::create(builder, loc, mlir::ValueRange{});
  builder.setInsertionPointAfter(remainderFor);

  return remainderFor;
}

static auto vectorizeMapLoop(mlir::scf::ForOp forOp, uint32_t width)
    -> mlir::LogicalResult {
  if (!hasVectorizableMapBody(forOp)) {
    return mlir::failure();
  }

  if (!isOne(forOp.getStep())) {
    return mlir::failure();
  }

  mlir::OpBuilder builder(forOp);
  mlir::Location loc = forOp.getLoc();

  mlir::Value widthValue =
      mlir::arith::ConstantIndexOp::create(builder, loc, width);

  mlir::Value tripCount = mlir::arith::SubIOp::create(
      builder, loc, forOp.getUpperBound(), forOp.getLowerBound());

  mlir::Value vectorTripCount =
      mlir::arith::DivSIOp::create(builder, loc, tripCount, widthValue);

  mlir::Value vectorElementCount =
      mlir::arith::MulIOp::create(builder, loc, vectorTripCount, widthValue);

  mlir::Value mainUb = mlir::arith::AddIOp::create(
      builder, loc, forOp.getLowerBound(), vectorElementCount);

  mlir::Value vectorStep =
      mlir::arith::MulIOp::create(builder, loc, forOp.getStep(), widthValue);

  auto vectorFor = mlir::scf::ForOp::create(builder, loc, forOp.getLowerBound(),
                                            mainUb, vectorStep);

  transferUnrollAttrsToVectorLoop(forOp, vectorFor);

  auto rollback = [&]() -> mlir::LogicalResult {
    vectorFor.erase();
    return mlir::failure();
  };

  clearLoopBody(vectorFor);
  builder.setInsertionPointToEnd(vectorFor.getBody());

  VectorBuildState state(builder, loc, forOp, vectorFor.getInductionVar(),
                         width);

  if (mlir::failed(vectorizeMapBody(state))) {
    return rollback();
  }

  mlir::scf::YieldOp::create(builder, loc, mlir::ValueRange{});
  builder.setInsertionPointAfter(vectorFor);

  buildScalarMapRemainderLoop(builder, loc, forOp, mainUb);

  forOp.erase();

  return mlir::success();
}

static auto hasOneFloatReduction(mlir::scf::ForOp forOp) -> bool {
  if (forOp.getNumRegionIterArgs() != 1 || forOp.getNumResults() != 1) {
    return false;
  }

  auto yieldOp = getYieldOp(forOp);
  if (!yieldOp || yieldOp.getNumOperands() != 1) {
    return false;
  }

  mlir::Type type = forOp.getRegionIterArg(0).getType();
  return type.isF32() || type.isF64();
}

static auto getReductionAdd(mlir::scf::ForOp forOp) -> mlir::arith::AddFOp {
  auto yieldOp = getYieldOp(forOp);
  if (!yieldOp || yieldOp.getNumOperands() != 1) {
    return {};
  }

  auto addOp = yieldOp.getOperand(0).getDefiningOp<mlir::arith::AddFOp>();
  if (!addOp) {
    return {};
  }

  mlir::Value acc = forOp.getRegionIterArg(0);
  if (addOp.getLhs() == acc || addOp.getRhs() == acc) {
    return addOp;
  }

  return {};
}

static auto getReductionExpr(mlir::scf::ForOp forOp, mlir::arith::AddFOp addOp)
    -> mlir::Value {
  mlir::Value acc = forOp.getRegionIterArg(0);
  return addOp.getLhs() == acc ? addOp.getRhs() : addOp.getLhs();
}

static auto buildVectorReductionMainLoop(
    mlir::OpBuilder &builder, mlir::Location loc, mlir::scf::ForOp oldFor,
    uint32_t width, mlir::Value mainUb, mlir::Value vectorStep,
    mlir::Value &reducedScalar) -> mlir::LogicalResult {
  mlir::Type elemType = oldFor.getResult(0).getType();

  mlir::Value zero = createZeroLike(builder, loc, elemType);
  if (!zero) {
    return mlir::failure();
  }

  auto vecType = mlir::VectorType::get({width}, elemType);
  mlir::Value zeroVec =
      mlir::vector::BroadcastOp::create(builder, loc, vecType, zero);

  auto vectorFor =
      mlir::scf::ForOp::create(builder, loc, oldFor.getLowerBound(), mainUb,
                               vectorStep, mlir::ValueRange{zeroVec});

  transferUnrollAttrsToVectorLoop(oldFor, vectorFor);

  auto rollback = [&]() -> mlir::LogicalResult {
    vectorFor.erase();
    return mlir::failure();
  };

  clearLoopBody(vectorFor);
  builder.setInsertionPointToEnd(vectorFor.getBody());

  auto addOp = getReductionAdd(oldFor);
  if (!addOp) {
    return rollback();
  }

  mlir::Value scalarExpr = getReductionExpr(oldFor, addOp);

  VectorBuildState state(builder, loc, oldFor, vectorFor.getInductionVar(),
                         width);

  mlir::Value vectorExpr = vectorizeValue(state, scalarExpr);
  if (!vectorExpr) {
    return rollback();
  }

  mlir::Value vectorAcc = vectorFor.getRegionIterArg(0);
  mlir::Value vectorSum =
      mlir::arith::AddFOp::create(builder, loc, vectorAcc, vectorExpr);

  mlir::scf::YieldOp::create(builder, loc, mlir::ValueRange{vectorSum});

  builder.setInsertionPointAfter(vectorFor);

  reducedScalar = mlir::vector::ReductionOp::create(
                      builder, loc, mlir::vector::CombiningKind::ADD,
                      vectorFor.getResult(0))
                      .getResult();

  return mlir::success();
}

static auto buildScalarReductionRemainderLoop(
    mlir::OpBuilder &builder, mlir::Location loc, mlir::scf::ForOp oldFor,
    mlir::Value lowerBound, mlir::Value initValue) -> mlir::scf::ForOp {
  auto oldYield = getYieldOp(oldFor);

  auto remainderFor =
      mlir::scf::ForOp::create(builder, loc, lowerBound, oldFor.getUpperBound(),
                               oldFor.getStep(), mlir::ValueRange{initValue});

  clearLoopBody(remainderFor);
  builder.setInsertionPointToEnd(remainderFor.getBody());

  mlir::IRMapping mapper;
  mapper.map(oldFor.getInductionVar(), remainderFor.getInductionVar());
  mapper.map(oldFor.getRegionIterArg(0), remainderFor.getRegionIterArg(0));

  for (mlir::Operation &op : oldFor.getBody()->without_terminator()) {
    builder.clone(op, mapper);
  }

  llvm::SmallVector<mlir::Value, 1> yieldValues;
  for (mlir::Value value : oldYield.getOperands()) {
    yieldValues.push_back(mapper.lookupOrDefault(value));
  }

  mlir::scf::YieldOp::create(builder, loc, yieldValues);

  builder.setInsertionPointAfter(remainderFor);
  return remainderFor;
}

static auto vectorizeReductionLoop(mlir::scf::ForOp forOp, uint32_t width)
    -> mlir::LogicalResult {
  if (!hasOneFloatReduction(forOp)) {
    return mlir::failure();
  }

  if (!isOne(forOp.getStep())) {
    return mlir::failure();
  }

  if (!getReductionAdd(forOp)) {
    return mlir::failure();
  }

  mlir::OpBuilder builder(forOp);
  mlir::Location loc = forOp.getLoc();

  mlir::Value widthValue =
      mlir::arith::ConstantIndexOp::create(builder, loc, width);

  mlir::Value tripCount = mlir::arith::SubIOp::create(
      builder, loc, forOp.getUpperBound(), forOp.getLowerBound());

  mlir::Value vectorTripCount =
      mlir::arith::DivSIOp::create(builder, loc, tripCount, widthValue);

  mlir::Value vectorElementCount =
      mlir::arith::MulIOp::create(builder, loc, vectorTripCount, widthValue);

  mlir::Value mainUb = mlir::arith::AddIOp::create(
      builder, loc, forOp.getLowerBound(), vectorElementCount);

  mlir::Value vectorStep =
      mlir::arith::MulIOp::create(builder, loc, forOp.getStep(), widthValue);

  mlir::Value reducedScalar;
  if (mlir::failed(buildVectorReductionMainLoop(
          builder, loc, forOp, width, mainUb, vectorStep, reducedScalar))) {
    return mlir::failure();
  }

  mlir::Value scalarAfterVector = mlir::arith::AddFOp::create(
      builder, loc, forOp.getInitArgs()[0], reducedScalar);

  auto remainderFor = buildScalarReductionRemainderLoop(
      builder, loc, forOp, mainUb, scalarAfterVector);

  forOp.getResult(0).replaceAllUsesWith(remainderFor.getResult(0));
  forOp.erase();

  return mlir::success();
}

struct LoopVectorizePass
    : public impl::LoopVectorizePassBase<LoopVectorizePass> {
  void runOnOperation() override {
    llvm::SmallVector<mlir::scf::ForOp, 8> loops;

    getOperation()->walk([&](mlir::scf::ForOp forOp) {
      if (forOp->hasAttr("vectorize")) {
        loops.push_back(forOp);
      }
    });

    for (mlir::scf::ForOp forOp : loops) {
      uint32_t width = 4;

      if (auto attr =
              forOp->getAttrOfType<mlir::IntegerAttr>("vectorize_width")) {
        width = static_cast<uint32_t>(attr.getInt());
      }

      if (width < 2) {
        forOp->removeAttr("vectorize");
        forOp->removeAttr("vectorize_width");
        continue;
      }

      if (mlir::succeeded(vectorizeMapLoop(forOp, width))) {
        continue;
      }

      if (mlir::succeeded(vectorizeReductionLoop(forOp, width))) {
        continue;
      }

      forOp->emitWarning("cmlir: loop vectorization skipped");
      forOp->removeAttr("vectorize");
      forOp->removeAttr("vectorize_width");
    }
  }
};

auto createLoopVectorizePass() -> std::unique_ptr<mlir::Pass> {
  return std::make_unique<LoopVectorizePass>();
}

} // namespace cmlir
