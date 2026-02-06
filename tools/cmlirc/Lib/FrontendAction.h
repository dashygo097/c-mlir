#ifndef CMLIRC_FRONTEND_ACTION_H
#define CMLIRC_FRONTEND_ACTION_H

#include "./ASTConsumer.h"
#include "./MLIRContextManager.h"
#include "clang/Frontend/FrontendActions.h"

namespace cmlirc {

class CMLIRCFrontendAction : public clang::ASTFrontendAction {
public:
  explicit CMLIRCFrontendAction()
      : mlir_context_manager_(std::make_unique<MLIRContextManager>()) {}
  ~CMLIRCFrontendAction() = default;

  std::unique_ptr<clang::ASTConsumer>
  CreateASTConsumer(clang::CompilerInstance &CI,
                    clang::StringRef file) override {
    return std::make_unique<CMLIRCASTConsumer>(&CI.getASTContext(),
                                               *mlir_context_manager_);
  }

  void EndSourceFileAction() override {
    llvm::outs() << "\nGenerated MLIR: \n";
    mlir_context_manager_->dump();
    llvm::outs() << "\n";
  }

private:
  std::unique_ptr<MLIRContextManager> mlir_context_manager_;
};

} // namespace cmlirc

#endif // CMLIRC_FRONTEND_ACTION_H
