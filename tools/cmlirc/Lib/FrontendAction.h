#ifndef CMLIRC_FRONTEND_ACTION_H
#define CMLIRC_FRONTEND_ACTION_H

#include "../ArgumentList.h"
#include "./ASTConsumer.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Frontend/FrontendAction.h"
#include "clang/Tooling/Tooling.h"

namespace cmlirc {

class CMLIRCFrontendAction : public clang::ASTFrontendAction {
public:
  explicit CMLIRCFrontendAction() {}
  ~CMLIRCFrontendAction() = default;

  std::unique_ptr<clang::ASTConsumer>
  CreateASTConsumer(clang::CompilerInstance &CI,
                    clang::StringRef file) override {
    context_manager_ = std::make_unique<ContextManager>(&CI.getASTContext());
    return std::make_unique<CMLIRCASTConsumer>(*context_manager_);
  }

  void EndSourceFileAction() override {
    if (options::Verbose)
      llvm::outs() << "\nGenerated MLIR: \n";

    mlir::PassManager pm(&context_manager_->MLIRContext());
    // mlir::OpPassManager &funcPM = pm.nest<mlir::func::FuncOp>();

    pm.addPass(mlir::createMem2Reg());
    if (options::FuncInline)
      pm.addPass(mlir::createInlinerPass());

    pm.addPass(mlir::createMem2Reg());
    if (options::SSCP)
      pm.addPass(mlir::createSCCPPass());

    pm.addPass(mlir::createMem2Reg());
    if (options::Canonicalize)
      pm.addPass(mlir::createCanonicalizerPass());

    pm.addPass(mlir::createMem2Reg());
    if (options::CSE)
      pm.addPass(mlir::createCSEPass());

    pm.addPass(mlir::createMem2Reg());
    if (options::LICM)
      pm.addPass(mlir::createLoopInvariantCodeMotionPass());

    pm.addPass(mlir::createMem2Reg());
    if (options::Canonicalize)
      pm.addPass(mlir::createCanonicalizerPass());

    pm.addPass(mlir::createMem2Reg());
    if (options::SymbolDCE)
      pm.addPass(mlir::createSymbolDCEPass());
    pm.addPass(mlir::createMem2Reg());

    if (mlir::failed(pm.run(context_manager_->Module()))) {
      llvm::errs() << "Failed to run optimization passes\n";
    }

    context_manager_->dump();
    llvm::outs() << "\n";
  }

private:
  std::unique_ptr<ContextManager> context_manager_;
};

} // namespace cmlirc

#endif // CMLIRC_FRONTEND_ACTION_H
