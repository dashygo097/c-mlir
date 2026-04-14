#ifndef CMLIRC_FRONTEND_ACTION_H
#define CMLIRC_FRONTEND_ACTION_H

#include "../ArgumentList.h"
#include "./Consumer.h"
#include "Pragmas/PragmaHandler.h"
#include "cmlir/Transforms/Passes.h"
#include "mlir/Dialect/Affine/Transforms/Passes.h"
#include "mlir/Dialect/Arith/Transforms/Passes.h"
#include "mlir/Dialect/Bufferization/Transforms/Passes.h"
#include "mlir/Dialect/Func/Extensions/InlinerExtension.h"
#include "mlir/Dialect/LLVMIR/Transforms/InlinerInterfaceImpl.h"
#include "mlir/Dialect/MemRef/Transforms/AllocationOpInterfaceImpl.h"
#include "mlir/Dialect/MemRef/Transforms/Passes.h"
#include "mlir/InitAllExtensions.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Frontend/FrontendAction.h"
#include "clang/Tooling/Tooling.h"

namespace cmlirc {

class CMLIRFrontendAction : public clang::ASTFrontendAction {
public:
  explicit CMLIRFrontendAction(llvm::raw_ostream *os = &llvm::outs())
      : outStream(os) {}
  ~CMLIRFrontendAction() override = default;

  auto CreateASTConsumer(clang::CompilerInstance &ci, clang::StringRef file)
      -> std::unique_ptr<clang::ASTConsumer> override {
    mlir::DialectRegistry registry;
    mlir::LLVM::registerInlinerInterface(registry);
    mlir::func::registerInlinerExtension(registry);
    mlir::memref::registerAllocationOpInterfaceExternalModels(registry);

    contextManager =
        std::make_unique<ContextManager>(&ci.getASTContext(), &registry);

    auto pragmaHandler = std::make_unique<CMLIRPragmaHandler>(loopHintMap);
    ci.getPreprocessor().AddPragmaHandler(pragmaHandler.release());

    return std::make_unique<CMLIRConsumer>(*contextManager, loopHintMap);
  }

  void EndSourceFileAction() override {
    mlir::PassManager pm(&contextManager->MLIRContext());
    pm.enableVerifier(false);

    pm.addPass(mlir::createCanonicalizerPass());
    pm.addPass(mlir::createSCCPPass());
    pm.addPass(cmlir::createConstPropPass());
    pm.addPass(cmlir::createArithCastPropPass());
    pm.addPass(mlir::arith::createIntRangeOptimizationsPass());
    pm.addPass(mlir::createCSEPass());

    if (options::funcInline) {
      pm.addPass(mlir::createInlinerPass());
      pm.addPass(mlir::createSymbolDCEPass());
      pm.addPass(mlir::createCanonicalizerPass());
      pm.addPass(mlir::createSCCPPass());
      pm.addPass(cmlir::createConstPropPass());
      pm.addPass(mlir::createCSEPass());
    }

    if (options::struct2Memref) {
      pm.addPass(cmlir::createStruct2MemrefPass());
    }
    pm.addPass(mlir::memref::createFoldMemRefAliasOpsPass());
    pm.addPass(mlir::memref::createNormalizeMemRefsPass());
    pm.addPass(mlir::createMem2Reg());
    pm.addPass(cmlir::createMem2RegPass());
    pm.addPass(mlir::createCanonicalizerPass());
    pm.addPass(mlir::createCSEPass());
    pm.addPass(mlir::createRemoveDeadValuesPass());

    if (options::raiseMemref2Affine || options::raiseSCF2Affine) {
      pm.addPass(cmlir::createRaiseMemref2AffinePass());
      pm.addNestedPass<mlir::func::FuncOp>(
          mlir::affine::createSimplifyAffineStructuresPass());
      pm.addNestedPass<mlir::func::FuncOp>(
          mlir::affine::createAffineScalarReplacementPass());
      pm.addNestedPass<mlir::func::FuncOp>(
          mlir::affine::createAffineLoopNormalizePass());
      pm.addPass(mlir::createCanonicalizerPass());
      pm.addPass(mlir::createCSEPass());
    }

    if (options::raiseSCF2Affine) {
      pm.addPass(cmlir::createRaiseSCF2AffinePass());
      pm.addNestedPass<mlir::func::FuncOp>(
          mlir::affine::createSimplifyAffineStructuresPass());
      pm.addNestedPass<mlir::func::FuncOp>(
          mlir::affine::createAffineScalarReplacementPass());
      pm.addNestedPass<mlir::func::FuncOp>(
          mlir::affine::createAffineLoopNormalizePass());
      pm.addPass(mlir::createCanonicalizerPass());
      pm.addPass(mlir::createCSEPass());
    }

    pm.addPass(mlir::createLoopInvariantCodeMotionPass());
    pm.addNestedPass<mlir::func::FuncOp>(
        mlir::bufferization::createBufferHoistingPass());
    pm.addPass(cmlir::createLoopUnrollPass());
    pm.addPass(mlir::createCanonicalizerPass());
    pm.addPass(cmlir::createLoopVectorizePass());
    if (options::fma) {
      pm.addPass(cmlir::createFMAPass());
    }
    pm.addPass(mlir::createLoopInvariantCodeMotionPass());

    pm.addPass(mlir::createMem2Reg());
    pm.addPass(cmlir::createMem2RegPass());
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

    if (options::disableOpt) {
      llvm::WithColor::warning()
          << "cmlirc: optimization passes are disabled\n";
      pm.clear();
    }

    if (mlir::failed(pm.run(contextManager->Module()))) {
      llvm::WithColor::error() << "cmlirc: failed to run optimization passes\n";
    }
    contextManager->dump(*outStream);
    outStream->flush();
  }

private:
  std::unique_ptr<ContextManager> contextManager;
  LoopHintMap loopHintMap;
  llvm::raw_ostream *outStream;
};

} // namespace cmlirc

#endif // CMLIRC_FRONTEND_ACTION_H
