#include "cmlir/Pipelines/Pipelines.h"
#include "cmlir/Support/Printing.h"
#include "cmlir/Transforms/Passes.h"
#include "mlir/Conversion/ControlFlowToSCF/ControlFlowToSCF.h"
#include "mlir/Conversion/SCFToAffine/SCFToAffine.h"
#include "mlir/Dialect/Affine/Transforms/Passes.h"
#include "mlir/Dialect/Arith/Transforms/Passes.h"
#include "mlir/Dialect/Bufferization/Transforms/Passes.h"
#include "mlir/Dialect/Func/Extensions/InlinerExtension.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/Transforms/InlinerInterfaceImpl.h"
#include "mlir/Dialect/Math/Transforms/Passes.h"
#include "mlir/Dialect/MemRef/IR/MemRefMemorySlot.h"
#include "mlir/Dialect/MemRef/Transforms/AllocationOpInterfaceImpl.h"
#include "mlir/Dialect/MemRef/Transforms/Passes.h"
#include "mlir/Dialect/SCF/Transforms/Passes.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"
#include "llvm/ADT/STLFunctionalExtras.h"

namespace cmlir {

namespace {

auto addValueCleanup(mlir::OpPassManager &pm) -> void {
  pm.addPass(mlir::createCanonicalizerPass());
  pm.addPass(mlir::createCSEPass());
}

auto addMetadataCleanup(mlir::OpPassManager &pm) -> void {
  pm.addPass(mlir::memref::createExpandStridedMetadataPass());
  addValueCleanup(pm);
}

auto addStorageCleanup(mlir::OpPassManager &pm) -> void {
  // Expose the original storage before splitting aggregates and promoting
  // slots. Constant propagation then sees SSA values instead of loads.
  pm.addPass(mlir::memref::createFoldMemRefAliasOpsPass());
  pm.addPass(mlir::createSROA());
  pm.addPass(mlir::createMem2Reg());
  pm.addPass(mlir::createSCCPPass());
  pm.addPass(cmlir::createArithFoldCastsPass());
  addValueCleanup(pm);
}

// Both tools use this sequence. The compiler executes checkpoints separately
// for stage diagnostics; an opt pipeline simply appends all the passes.
auto populatePipeline(
    mlir::OpPassManager &pm, const PipelineOptions &options,
    llvm::function_ref<mlir::LogicalResult(llvm::StringRef)> finishStage)
    -> mlir::LogicalResult {
  addMetadataCleanup(pm);
  addStorageCleanup(pm);
  if (mlir::failed(finishStage("SSA cleanup"))) {
    return mlir::failure();
  }

  if (options.inlineFunctions) {
    pm.addPass(mlir::createInlinerPass());
    // Actual argument shapes can expose new metadata and storage folds.
    addMetadataCleanup(pm);
    addStorageCleanup(pm);
    pm.addPass(mlir::createSymbolDCEPass());
    if (mlir::failed(finishStage("inlining"))) {
      return mlir::failure();
    }
  }

  if (options.promoteStructs) {
    pm.addPass(cmlir::createStruct2MemrefPass());
    addStorageCleanup(pm);
    if (mlir::failed(finishStage("struct promotion"))) {
      return mlir::failure();
    }
  }

  if (options.liftCFToSCF) {
    // Inlining can introduce new CFG edges. Structure them before loop passes;
    // directly lowered SCF loops and their pragma attributes remain intact.
    pm.addPass(mlir::createLiftControlFlowToSCFPass());
    addValueCleanup(pm);
    if (mlir::failed(finishStage("control flow lifting"))) {
      return mlir::failure();
    }
  }

  if (options.upliftToFMA) {
    pm.addPass(mlir::math::createMathUpliftToFMA());
  }
  // Simplify cross-dialect loop operations, then fold induction arithmetic
  // into the range before unrolling, vectorization and optional Affine raising.
  pm.addPass(mlir::createSCFForLoopCanonicalizationPass());
  pm.addPass(mlir::createForLoopRangeFoldingPass());
  addValueCleanup(pm);
  pm.addPass(mlir::createLoopInvariantCodeMotionPass());
  pm.addNestedPass<mlir::func::FuncOp>(
      mlir::bufferization::createBufferHoistingPass());
  if (mlir::failed(finishStage("loop preparation"))) {
    return mlir::failure();
  }

  // Consume SCF unroll hints before vectorization uses Affine temporarily.
  pm.addPass(cmlir::createLoopUnrollPass());
  pm.addPass(cmlir::createLoopVectorizePass());
  addValueCleanup(pm);
  if (mlir::failed(finishStage("loop transformations"))) {
    return mlir::failure();
  }

  // Pragma transformations have already run; upstream decides which loops
  // can be represented in Affine.
  if (options.raiseSCFToAffine || options.raiseMemRefToAffine) {
    pm.addPass(mlir::memref::createNormalizeMemRefsPass());
    if (options.raiseSCFToAffine) {
      pm.addPass(mlir::createRaiseSCFToAffinePass());
    }
    if (options.raiseMemRefToAffine) {
      pm.addNestedPass<mlir::func::FuncOp>(
          mlir::affine::createRaiseMemrefToAffine());
    }
    pm.addNestedPass<mlir::func::FuncOp>(
        mlir::affine::createAffineLoopNormalizePass());
    pm.addNestedPass<mlir::func::FuncOp>(
        mlir::affine::createAffineScalarReplacementPass());
    pm.addPass(mlir::affine::createLoopFusionPass());
    pm.addNestedPass<mlir::func::FuncOp>(
        mlir::affine::createSimplifyAffineStructuresPass());
    addValueCleanup(pm);
    if (mlir::failed(finishStage("affine transformations"))) {
      return mlir::failure();
    }
  }

  pm.addPass(mlir::createLoopInvariantCodeMotionPass());

  pm.addNestedPass<mlir::func::FuncOp>(
      mlir::bufferization::createBufferLoopHoistingPass());
  pm.addPass(mlir::createControlFlowSinkPass());
  addMetadataCleanup(pm);
  addStorageCleanup(pm);
  pm.addPass(mlir::arith::createIntRangeOptimizationsPass());
  addValueCleanup(pm);
  pm.addPass(mlir::createRemoveDeadValuesPass());
  pm.addPass(mlir::createSymbolDCEPass());
  return finishStage("final cleanup");
}

} // namespace

auto registerPipelineInterfaces(mlir::DialectRegistry &registry) -> void {
  mlir::LLVM::registerInlinerInterface(registry);
  mlir::func::registerInlinerExtension(registry);
  mlir::memref::registerAllocationOpInterfaceExternalModels(registry);
  mlir::memref::registerMemorySlotExternalModels(registry);
}

auto buildDefaultPipeline(mlir::OpPassManager &pm,
                          const PipelineOptions &options) -> void {
  (void)populatePipeline(
      pm, options,
      [](llvm::StringRef) -> mlir::LogicalResult { return mlir::success(); });
}

auto runPipeline(mlir::ModuleOp module, const PipelineOptions &options)
    -> llvm::Error {
  if (mlir::failed(mlir::verify(module))) {
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "invalid module before optimization");
  }
  if (options.dumpStages) {
    llvm::errs() << "// clang-to-mlir stage: AST lowering\n";
    cmlir::printIR(module, llvm::errs());
    llvm::errs() << '\n';
  }
  if (!options.optimize) {
    return llvm::Error::success();
  }
  mlir::DialectRegistry registry;
  registerPipelineInterfaces(registry);
  module.getContext()->appendDialectRegistry(registry);
  mlir::PassManager pm(module.getContext());
  pm.enableVerifier(true);
  if (options.timingScope) {
    pm.enableTiming(*options.timingScope);
  }
  llvm::StringRef failedStage;
  auto result = populatePipeline(
      pm, options, [&](llvm::StringRef name) -> mlir::LogicalResult {
        if (mlir::failed(pm.run(module))) {
          failedStage = name;
          return mlir::failure();
        }
        if (options.dumpStages) {
          llvm::errs() << "// clang-to-mlir stage: " << name << '\n';
          cmlir::printIR(module, llvm::errs());
          llvm::errs() << '\n';
        }
        pm.clear();
        return mlir::success();
      });
  if (mlir::failed(result)) {
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "pipeline stage '%s' failed",
                                   failedStage.str().c_str());
  }
  return llvm::Error::success();
}

} // namespace cmlir
