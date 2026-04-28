#ifndef CHWC_FRONTEND_ACTION_H
#define CHWC_FRONTEND_ACTION_H

#include "../ArgumentList.h"
#include "./Consumer.h"
#include "mlir/Pass/PassManager.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Frontend/FrontendAction.h"
#include "clang/Tooling/Tooling.h"
#include "llvm/Support/WithColor.h"

namespace chwc {

class CHWFrontendAction : public clang::ASTFrontendAction {
public:
  explicit CHWFrontendAction(llvm::raw_ostream *os = &llvm::outs())
      : outStream(os) {}
  ~CHWFrontendAction() override = default;

  auto CreateASTConsumer(clang::CompilerInstance &ci, clang::StringRef file)
      -> std::unique_ptr<clang::ASTConsumer> override {
    mlir::DialectRegistry registry;

    contextManager =
        std::make_unique<ContextManager>(&ci.getASTContext(), &registry);

    return std::make_unique<CHWConsumer>(*contextManager);
  }

  void EndSourceFileAction() override {
    mlir::PassManager pm(&contextManager->MLIRContext());
    pm.enableVerifier(true);

    if (options::disableOpt) {
      llvm::WithColor::warning() << "chwc: optimization passes are disabled\n";
      pm.clear();
    }

    if (mlir::failed(pm.run(contextManager->Module()))) {
      llvm::WithColor::error() << "chwc: failed to run optimization passes\n";
    }
    contextManager->dump(*outStream);
    outStream->flush();
  }

private:
  std::unique_ptr<ContextManager> contextManager;
  llvm::raw_ostream *outStream;
};

} // namespace chwc

#endif // CHWC_FRONTEND_ACTION_H
