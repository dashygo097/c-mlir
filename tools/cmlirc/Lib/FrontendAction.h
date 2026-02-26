#ifndef CMLIRC_FRONTEND_ACTION_H
#define CMLIRC_FRONTEND_ACTION_H

#include "../ArgumentList.h"
#include "./Consumer.h"
#include "Pragmas/PragmaHandler.h"
#include "cmlir/Transforms/Passes.h"
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
    context_manager_ = std::make_unique<ContextManager>(&CI.getASTContext());

    auto pragma_handler = std::make_unique<CMLIRPragmaHandler>(loop_hints_);
    CI.getPreprocessor().AddPragmaHandler(pragma_handler.release());

    return std::make_unique<CMLIRConsumer>(*context_manager_, loop_hints_);
  }

  void EndSourceFileAction() override {
    if (options::Verbose)
      llvm::outs() << "\nGenerated MLIR: \n";

    mlir::PassManager pm(&context_manager_->MLIRContext());

    if (options::EnableLoopUnroll)
      pm.addNestedPass<mlir::func::FuncOp>(cmlir::createLoopUnrollPass());

    if (options::Canonicalize)
      pm.addPass(mlir::createCanonicalizerPass());

    if (options::CSE)
      pm.addPass(mlir::createCSEPass());

    if (options::FuncInline)
      pm.addPass(mlir::createInlinerPass());

    if (options::ConstProp)
      pm.addPass(cmlir::createConstPropPass());

    if (options::SSCP)
      pm.addPass(mlir::createSCCPPass());

    if (options::Struct2Memref)
      pm.addPass(cmlir::createStruct2MemrefPass());

    if (options::Mem2Reg) {
      pm.addPass(mlir::createMem2Reg());
      pm.addPass(cmlir::createMem2RegPass());
    }

    if (options::ConstProp)
      pm.addPass(cmlir::createConstPropPass());

    if (options::RaiseSCF2Affine)
      pm.addPass(cmlir::createRaiseSCF2AffinePass());

    if (options::FMA)
      pm.addPass(cmlir::createFMAPass());

    if (options::Canonicalize)
      pm.addPass(mlir::createCanonicalizerPass());

    if (options::CSE)
      pm.addPass(mlir::createCSEPass());

    if (options::LICM)
      pm.addPass(mlir::createLoopInvariantCodeMotionPass());

    if (options::Canonicalize)
      pm.addPass(mlir::createCanonicalizerPass());

    if (options::CSE)
      pm.addPass(mlir::createCSEPass());

    if (options::SymbolDCE)
      pm.addPass(mlir::createSymbolDCEPass());

    if (mlir::failed(pm.run(context_manager_->Module())))
      llvm::errs() << "Failed to run optimization passes\n";

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
