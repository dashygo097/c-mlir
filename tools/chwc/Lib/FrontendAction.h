#ifndef CHWC_FRONTEND_ACTION_H
#define CHWC_FRONTEND_ACTION_H

#include "../ArgumentList.h"
#include "./Consumer.h"
#include "circt/Dialect/Comb/CombDialect.h"
#include "circt/Dialect/HW/HWDialect.h"
#include "circt/Dialect/SV/SVDialect.h"
#include "circt/Dialect/Seq/SeqDialect.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"
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

    registry.insert<mlir::arith::ArithDialect>();
    registry.insert<circt::hw::HWDialect>();
    registry.insert<circt::comb::CombDialect>();
    registry.insert<circt::seq::SeqDialect>();
    registry.insert<circt::sv::SVDialect>();

    contextManager =
        std::make_unique<CHWContextManager>(&ci.getASTContext(), &registry);

    contextManager->MLIRContext().loadDialect<mlir::arith::ArithDialect>();
    contextManager->MLIRContext().loadDialect<circt::hw::HWDialect>();
    contextManager->MLIRContext().loadDialect<circt::comb::CombDialect>();
    contextManager->MLIRContext().loadDialect<circt::seq::SeqDialect>();
    contextManager->MLIRContext().loadDialect<circt::sv::SVDialect>();

    return std::make_unique<CHWConsumer>(*contextManager);
  }

  void EndSourceFileAction() override {
    mlir::PassManager pm(&contextManager->MLIRContext());
    pm.enableVerifier(true);

    if (options::disableOpt) {
      llvm::WithColor::warning() << "chwc: optimization passes are disabled\n";
    } else {
      pm.addPass(mlir::createCanonicalizerPass());
      pm.addPass(mlir::createCSEPass());
      pm.addPass(mlir::createSymbolDCEPass());

      pm.addPass(mlir::createCanonicalizerPass());
      pm.addPass(mlir::createCSEPass());
    }

    if (mlir::failed(pm.run(contextManager->Module()))) {
      llvm::WithColor::error() << "chwc: failed to run optimization passes\n";
      return;
    }

    contextManager->dump(*outStream);
    outStream->flush();
  }

private:
  std::unique_ptr<CHWContextManager> contextManager;
  llvm::raw_ostream *outStream;
};

} // namespace chwc

#endif // CHWC_FRONTEND_ACTION_H
