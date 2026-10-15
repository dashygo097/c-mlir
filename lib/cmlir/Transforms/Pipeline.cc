#include "cmlir/Transforms/Pipeline.h"
#include "cmlir/Support/Printing.h"
#include "cmlir/Transforms/Passes.h"
#include "mlir/Dialect/Affine/Transforms/Passes.h"
#include "mlir/Dialect/Arith/Transforms/Passes.h"
#include "mlir/Dialect/Bufferization/Transforms/Passes.h"
#include "mlir/Dialect/Func/Extensions/InlinerExtension.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/Transforms/InlinerInterfaceImpl.h"
#include "mlir/Dialect/MemRef/IR/MemRefMemorySlot.h"
#include "mlir/Dialect/MemRef/Transforms/AllocationOpInterfaceImpl.h"
#include "mlir/Dialect/MemRef/Transforms/Passes.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"

namespace cmlir {

namespace {

auto addValueCleanup(mlir::PassManager &pm) -> void {
  pm.addPass(mlir::createCanonicalizerPass());
  pm.addPass(mlir::createCSEPass());
}

auto addMetadataCleanup(mlir::PassManager &pm) -> void {
  pm.addPass(mlir::memref::createExpandStridedMetadataPass());
  addValueCleanup(pm);
}

auto addStorageCleanup(mlir::PassManager &pm) -> void {
  // Expose the original storage before splitting aggregates and promoting
  // slots. Constant propagation then sees SSA values instead of loads.
  pm.addPass(mlir::memref::createFoldMemRefAliasOpsPass());
  pm.addPass(mlir::createSROA());
  pm.addPass(mlir::createMem2Reg());
  pm.addPass(mlir::createSCCPPass());
  pm.addPass(cmlir::createArithCastPropPass());
  addValueCleanup(pm);
}

auto runStage(mlir::PassManager &pm, mlir::ModuleOp module,
              llvm::StringRef name, bool dump) -> llvm::Error {
  if (mlir::failed(pm.run(module))) {
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "pipeline stage '%s' failed",
                                   name.str().c_str());
  }
  if (dump) {
    llvm::errs() << "// clang-to-mlir stage: " << name << '\n';
    cmlir::printIR(module, llvm::errs());
    llvm::errs() << '\n';
  }
  pm.clear();
  return llvm::Error::success();
}

} // namespace

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
  mlir::LLVM::registerInlinerInterface(registry);
  mlir::func::registerInlinerExtension(registry);
  mlir::memref::registerAllocationOpInterfaceExternalModels(registry);
  mlir::memref::registerMemorySlotExternalModels(registry);
  module.getContext()->appendDialectRegistry(registry);
  mlir::PassManager pm(module.getContext());
  pm.enableVerifier(true);
  if (options.timingScope) {
    pm.enableTiming(*options.timingScope);
  }

  addMetadataCleanup(pm);
  addStorageCleanup(pm);
  if (auto error = runStage(pm, module, "SSA cleanup", options.dumpStages)) {
    return error;
  }

  if (options.inlineFunctions) {
    pm.addPass(mlir::createInlinerPass());
    // Actual argument shapes can expose new metadata and storage folds.
    addMetadataCleanup(pm);
    addStorageCleanup(pm);
    pm.addPass(mlir::createSymbolDCEPass());
    if (auto error = runStage(pm, module, "inlining", options.dumpStages)) {
      return error;
    }
  }

  if (options.promoteStructs) {
    pm.addPass(cmlir::createStruct2MemrefPass());
    addStorageCleanup(pm);
    if (auto error =
            runStage(pm, module, "struct promotion", options.dumpStages)) {
      return error;
    }
  }

  if (options.fuseMultiplyAdd) {
    pm.addPass(cmlir::createFMAPass());
  }
  pm.addPass(mlir::createLoopInvariantCodeMotionPass());
  pm.addNestedPass<mlir::func::FuncOp>(
      mlir::bufferization::createBufferHoistingPass());
  if (auto error =
          runStage(pm, module, "loop preparation", options.dumpStages)) {
    return error;
  }

  // Analyze and fuse eligible loops before code expansion. Explicit pragma
  // loops stay in SCF for the transformations that understand their hints.
  if (options.raiseSCFToAffine || options.raiseMemRefToAffine) {
    pm.addPass(mlir::memref::createNormalizeMemRefsPass());
    if (options.raiseSCFToAffine) {
      pm.addPass(cmlir::createRaiseSCF2AffinePass());
    }
    if (options.raiseMemRefToAffine) {
      pm.addPass(cmlir::createRaiseMemref2AffinePass());
    }
    pm.addNestedPass<mlir::func::FuncOp>(
        mlir::affine::createAffineLoopNormalizePass());
    pm.addNestedPass<mlir::func::FuncOp>(
        mlir::affine::createAffineScalarReplacementPass());
    pm.addPass(mlir::affine::createLoopFusionPass());
    pm.addNestedPass<mlir::func::FuncOp>(
        mlir::affine::createSimplifyAffineStructuresPass());
    addValueCleanup(pm);
    if (auto error = runStage(pm, module, "affine transformations",
                              options.dumpStages)) {
      return error;
    }
  }

  pm.addPass(cmlir::createLoopVectorizePass());
  pm.addPass(mlir::createCanonicalizerPass());
  pm.addPass(cmlir::createLoopUnrollPass());
  pm.addPass(mlir::createLoopInvariantCodeMotionPass());
  if (auto error =
          runStage(pm, module, "loop transformations", options.dumpStages)) {
    return error;
  }

  pm.addNestedPass<mlir::func::FuncOp>(
      mlir::bufferization::createBufferLoopHoistingPass());
  pm.addPass(mlir::createControlFlowSinkPass());
  addMetadataCleanup(pm);
  addStorageCleanup(pm);
  pm.addPass(mlir::arith::createIntRangeOptimizationsPass());
  addValueCleanup(pm);
  pm.addPass(mlir::createRemoveDeadValuesPass());
  pm.addPass(mlir::createSymbolDCEPass());
  return runStage(pm, module, "final cleanup", options.dumpStages);
}

} // namespace cmlir
