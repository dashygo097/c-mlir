#ifndef CLANG_TO_MLIR_FRONTEND_ACTION_H
#define CLANG_TO_MLIR_FRONTEND_ACTION_H

#include "cmlir/Frontend/Frontend.h"
#include "cmlir/Pipelines/Pipelines.h"
#include "mlir/IR/MLIRContext.h"
#include "clang/Frontend/FrontendAction.h"
#include "clang/Tooling/Tooling.h"

namespace cmlir::frontend {

class FrontendAction : public clang::ASTFrontendAction {
public:
  FrontendAction(llvm::raw_ostream &out, CodeGenOptions codegen,
                 PipelineOptions pipeline)
      : out(out), codegen(std::move(codegen)), pipeline(pipeline) {}

  auto CreateASTConsumer(clang::CompilerInstance &, llvm::StringRef)
      -> std::unique_ptr<clang::ASTConsumer> override;

private:
  llvm::raw_ostream &out;
  CodeGenOptions codegen;
  PipelineOptions pipeline;
  mlir::MLIRContext ir;
  LoopHintMap hints;
};

class ActionFactory : public clang::tooling::FrontendActionFactory {
public:
  ActionFactory(llvm::raw_ostream &out, CodeGenOptions codegen,
                PipelineOptions pipeline)
      : out(out), codegen(std::move(codegen)), pipeline(pipeline) {}

  auto create() -> std::unique_ptr<clang::FrontendAction> override {
    return std::make_unique<FrontendAction>(out, codegen, pipeline);
  }

private:
  llvm::raw_ostream &out;
  CodeGenOptions codegen;
  PipelineOptions pipeline;
};

} // namespace cmlir::frontend

#endif // CLANG_TO_MLIR_FRONTEND_ACTION_H
