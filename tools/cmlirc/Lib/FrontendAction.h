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
  explicit CMLIRFrontendAction(llvm::raw_ostream *outputStream = &llvm::outs())
      : output_stream_(outputStream) {}
  ~CMLIRFrontendAction() = default;

  std::unique_ptr<clang::ASTConsumer>
  CreateASTConsumer(clang::CompilerInstance &CI,
                    clang::StringRef file) override {
    mlir::DialectRegistry registry;
    mlir::LLVM::registerInlinerInterface(registry);
    mlir::func::registerInlinerExtension(registry);
    mlir::memref::registerAllocationOpInterfaceExternalModels(registry);

    context_manager_ =
        std::make_unique<ContextManager>(&CI.getASTContext(), &registry);

    auto pragma_handler = std::make_unique<CMLIRPragmaHandler>(loop_hints_);
    CI.getPreprocessor().AddPragmaHandler(pragma_handler.release());

    return std::make_unique<CMLIRConsumer>(*context_manager_, loop_hints_);
  }

  void EndSourceFileAction() override {
    mlir::PassManager pm(&context_manager_->MLIRContext());
    pm.enableVerifier(false);

    pm.addPass(mlir::createCanonicalizerPass());
    pm.addPass(mlir::createSCCPPass());
    pm.addPass(cmlir::createConstPropPass());
    pm.addPass(mlir::arith::createIntRangeOptimizationsPass());
    pm.addPass(mlir::createCSEPass());

    if (options::FuncInline) {
      pm.addPass(mlir::createInlinerPass());
      pm.addPass(mlir::createSymbolDCEPass());
      pm.addPass(mlir::createCanonicalizerPass());
      pm.addPass(mlir::createSCCPPass());
      pm.addPass(cmlir::createConstPropPass());
      pm.addPass(mlir::createCSEPass());
    }

    if (options::Struct2Memref)
      pm.addPass(cmlir::createStruct2MemrefPass());
    pm.addPass(mlir::memref::createFoldMemRefAliasOpsPass());
    pm.addPass(mlir::memref::createNormalizeMemRefsPass());
    pm.addPass(mlir::createMem2Reg());
    pm.addPass(cmlir::createMem2RegPass());
    pm.addPass(mlir::createCanonicalizerPass());
    pm.addPass(mlir::createCSEPass());
    pm.addPass(mlir::createRemoveDeadValuesPass());

    if (options::RaiseSCF2Affine) {
      pm.addPass(cmlir::createRaiseSCF2AffinePass());
      pm.addPass(mlir::affine::createSimplifyAffineStructuresPass());
      pm.addPass(mlir::affine::createAffineScalarReplacementPass());
      pm.addPass(mlir::affine::createAffineLoopNormalizePass());
      pm.addPass(mlir::createCanonicalizerPass());
      pm.addPass(mlir::createCSEPass());
    }

    pm.addPass(mlir::createLoopInvariantCodeMotionPass());
    pm.addNestedPass<mlir::func::FuncOp>(
        mlir::bufferization::createBufferHoistingPass());
    pm.addPass(cmlir::createLoopUnrollPass());
    pm.addPass(mlir::createCanonicalizerPass());
    pm.addPass(cmlir::createLoopVectorizePass());
    if (options::FMA)
      pm.addPass(cmlir::createFMAPass());
    pm.addPass(mlir::createLoopInvariantCodeMotionPass());

    pm.addPass(mlir::createMem2Reg());
    pm.addPass(cmlir::createMem2RegPass());
    pm.addPass(mlir::memref::createFoldMemRefAliasOpsPass());
    pm.addNestedPass<mlir::func::FuncOp>(
        mlir::bufferization::createBufferLoopHoistingPass());

    pm.addPass(mlir::createControlFlowSinkPass());
    pm.addPass(cmlir::createConstPropPass());
    pm.addPass(mlir::arith::createIntRangeOptimizationsPass());
    pm.addPass(mlir::createCanonicalizerPass());
    pm.addPass(mlir::createCSEPass());
    pm.addPass(mlir::createRemoveDeadValuesPass());
    pm.addPass(mlir::createSymbolDCEPass());
    pm.addPass(mlir::createTopologicalSortPass());

    if (options::DisableOpt) {
      llvm::WithColor::warning()
          << "cmlirc: optimization passes are disabled\n";
      pm.clear();
    }

    if (mlir::failed(pm.run(context_manager_->Module())))
      llvm::WithColor::error() << "cmlirc: failed to run optimization passes\n";
    context_manager_->dump(*output_stream_);
    output_stream_->flush();
  }

private:
  std::unique_ptr<ContextManager> context_manager_;
  LoopHintMap loop_hints_;
  llvm::raw_ostream *output_stream_;
};

} // namespace cmlirc

#endif // CMLIRC_FRONTEND_ACTION_H
