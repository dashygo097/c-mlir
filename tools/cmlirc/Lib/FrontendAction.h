#ifndef CMLIRC_FRONTEND_ACTION_H
#define CMLIRC_FRONTEND_ACTION_H

#include "./ASTConsumer.h"
#include "clang/Frontend/FrontendActions.h"

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
    llvm::outs() << "\nGenerated MLIR: \n";
    context_manager_->dump();
    llvm::outs() << "\n";
  }

private:
  std::unique_ptr<ContextManager> context_manager_;
};

} // namespace cmlirc

#endif // CMLIRC_FRONTEND_ACTION_H
