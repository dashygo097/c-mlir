#include "FrontendAction.h"
#include "Pragmas/PragmaHandler.h"
#include "cmlir/Support/Printing.h"
#include "clang/AST/ASTConsumer.h"
#include "clang/Frontend/CompilerInstance.h"

namespace cmlir::frontend {

namespace {

class Consumer final : public clang::ASTConsumer {
public:
  Consumer(llvm::raw_ostream &out, mlir::MLIRContext &ir,
           const CodeGenOptions &codegen, const PipelineOptions &pipeline,
           const LoopHintMap &hints)
      : out(out), ir(ir), codegen(codegen), pipeline(pipeline), hints(hints) {}

  auto HandleTranslationUnit(clang::ASTContext &ast) -> void override {
    if (ast.getDiagnostics().hasErrorOccurred()) {
      return;
    }
    auto module = generateModule(ast, ir, codegen, hints);
    if (!module) {
      report(ast, llvm::toString(module.takeError()));
      return;
    }
    if (auto error = cmlir::runPipeline(module->get(), pipeline)) {
      report(ast, llvm::toString(std::move(error)));
      return;
    }
    cmlir::printIR(module->get(), out);
    out << '\n';
    out.flush();
  }

private:
  static auto report(clang::ASTContext &ast, llvm::StringRef message) -> void {
    auto &diags = ast.getDiagnostics();
    diags.Report(diags.getCustomDiagID(clang::DiagnosticsEngine::Error, "%0"))
        << message;
  }

  llvm::raw_ostream &out;
  mlir::MLIRContext &ir;
  const CodeGenOptions &codegen;
  const PipelineOptions &pipeline;
  const LoopHintMap &hints;
};

} // namespace

auto FrontendAction::CreateASTConsumer(clang::CompilerInstance &ci,
                                       llvm::StringRef)
    -> std::unique_ptr<clang::ASTConsumer> {
  ci.getPreprocessor().AddPragmaHandler(new CMLIRPragmaHandler(hints));
  return std::make_unique<Consumer>(out, ir, codegen, pipeline, hints);
}

} // namespace cmlir::frontend
