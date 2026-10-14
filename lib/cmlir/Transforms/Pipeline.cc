#include "cmlir/Transforms/Pipeline.h"
#include "cmlir/Transforms/Passes.h"
#include "mlir/Dialect/Affine/Transforms/Passes.h"
#include "mlir/Dialect/Arith/Transforms/Passes.h"
#include "mlir/Dialect/Bufferization/Transforms/Passes.h"
#include "mlir/Dialect/Func/Extensions/InlinerExtension.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/Transforms/InlinerInterfaceImpl.h"
#include "mlir/Dialect/MemRef/Transforms/AllocationOpInterfaceImpl.h"
#include "mlir/Dialect/MemRef/Transforms/Passes.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"

namespace cmlir {

namespace {

auto runStage(mlir::PassManager &pm, mlir::ModuleOp module,
              llvm::StringRef name, bool dump) -> llvm::Error {
  if (mlir::failed(pm.run(module))) {
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "pipeline stage '%s' failed",
                                   name.str().c_str());
  }
  if (dump) {
    llvm::errs() << "// clang-to-mlir stage: " << name << '\n';
    module.print(llvm::errs());
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
    module.print(llvm::errs());
    llvm::errs() << '\n';
  }
  if (!options.optimize) {
    return llvm::Error::success();
  }
  mlir::DialectRegistry registry;
  mlir::LLVM::registerInlinerInterface(registry);
  mlir::func::registerInlinerExtension(registry);
  mlir::memref::registerAllocationOpInterfaceExternalModels(registry);
  module.getContext()->appendDialectRegistry(registry);
  mlir::PassManager pm(module.getContext());
  pm.enableVerifier(true);

  // PHASE 1: INITIAL CLEANUP & SSA CONVERSION
  pm.addPass(mlir::createCanonicalizerPass());
  pm.addPass(mlir::createSCCPPass());
  pm.addPass(cmlir::createConstPropPass());
  pm.addPass(mlir::createCSEPass());

  pm.addPass(mlir::memref::createFoldMemRefAliasOpsPass());
  pm.addPass(mlir::memref::createNormalizeMemRefsPass());
  pm.addPass(mlir::createMem2Reg());
  pm.addPass(mlir::createCanonicalizerPass());
  pm.addPass(mlir::createCSEPass());

  if (auto error = runStage(pm, module, "SSA cleanup", options.dumpStages)) {
    return error;
  }

  // PHASE 2: INLINING
  if (options.inlineFunctions) {
    pm.addPass(mlir::createInlinerPass());
    pm.addPass(mlir::createSymbolDCEPass());
    pm.addPass(mlir::createCanonicalizerPass());
    pm.addPass(mlir::createSCCPPass());
    pm.addPass(cmlir::createConstPropPass());
    pm.addPass(mlir::createCSEPass());

    pm.addPass(mlir::createMem2Reg());
    pm.addPass(mlir::createCanonicalizerPass());
  }

  if (auto error = runStage(pm, module, "inlining", options.dumpStages)) {
    return error;
  }

  // PHASE 3: STRUCT OPTIMIZATIONS
  if (options.promoteStructs) {
    pm.addPass(cmlir::createStruct2MemrefPass());
    pm.addPass(mlir::createMem2Reg());
    pm.addPass(mlir::createCanonicalizerPass());
  }

  if (auto error =
          runStage(pm, module, "struct promotion", options.dumpStages)) {
    return error;
  }

  // PHASE 4: LOOP OPTIMIZATIONS
  pm.addPass(mlir::createLoopInvariantCodeMotionPass());
  pm.addNestedPass<mlir::func::FuncOp>(
      mlir::bufferization::createBufferHoistingPass());

  // FMA fusion
  if (options.fuseMultiplyAdd) {
    pm.addPass(cmlir::createFMAPass());
  }

  // Loop vectorization
  pm.addPass(cmlir::createLoopVectorizePass());
  pm.addPass(mlir::createCanonicalizerPass());

  // Loop unrolling
  pm.addPass(cmlir::createLoopUnrollPass());
  pm.addPass(mlir::createLoopInvariantCodeMotionPass());

  if (auto error =
          runStage(pm, module, "loop transformations", options.dumpStages)) {
    return error;
  }

  // PHASE 5: AFFINE TRANSFORMATIONS
  if (options.raiseSCFToAffine || options.raiseMemRefToAffine) {
    if (options.raiseSCFToAffine) {
      pm.addPass(cmlir::createRaiseSCF2AffinePass());
    }

    pm.addPass(mlir::createCanonicalizerPass());
    pm.addPass(mlir::createCSEPass());

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

    pm.addPass(mlir::createCanonicalizerPass());
    pm.addPass(mlir::createCSEPass());
  }

  if (auto error =
          runStage(pm, module, "affine transformations", options.dumpStages)) {
    return error;
  }

  // PHASE 6: FINAL CLEANUP
  pm.addPass(mlir::createMem2Reg());
  pm.addPass(mlir::memref::createFoldMemRefAliasOpsPass());
  pm.addNestedPass<mlir::func::FuncOp>(
      mlir::bufferization::createBufferLoopHoistingPass());

  pm.addPass(mlir::createControlFlowSinkPass());
  pm.addPass(cmlir::createConstPropPass());
  pm.addPass(cmlir::createArithCastPropPass());
  pm.addPass(mlir::arith::createIntRangeOptimizationsPass());

  pm.addPass(mlir::createCanonicalizerPass());
  pm.addPass(mlir::createCSEPass());
  pm.addPass(mlir::createRemoveDeadValuesPass());
  pm.addPass(mlir::createSymbolDCEPass());
  pm.addPass(mlir::createTopologicalSortPass());

  return runStage(pm, module, "final cleanup", options.dumpStages);
}

} // namespace cmlir
