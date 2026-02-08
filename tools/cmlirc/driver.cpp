#include "./ArgumentList.h"
#include "Lib/FrontendAction.h"
#include "clang/Tooling/CommonOptionsParser.h"
#include "clang/Tooling/Tooling.h"
#include "llvm/Support/CommandLine.h"

using namespace clang;
using namespace clang::tooling;
using namespace cmlirc;

namespace cmlirc::options {
llvm::cl::OptionCategory toolOptions("CMLIRC Options");

llvm::cl::opt<bool> Verbose("v", llvm::cl::init(false),
                            llvm::cl::desc("Enable verbose"),
                            llvm::cl::cat(toolOptions));

llvm::cl::opt<bool> MergeConstants(
    "merge-consts", llvm::cl::init(true),
    llvm::cl::desc("Merge constant operations in the generated MLIR"),
    llvm::cl::cat(toolOptions));

llvm::cl::extrahelp CommonHelp(CommonOptionsParser::HelpMessage);

} // namespace cmlirc::options

int main(int argc, const char **argv) {
  auto ExpectedParser =
      CommonOptionsParser::create(argc, argv, options::toolOptions);

  if (!ExpectedParser) {
    llvm::errs() << ExpectedParser.takeError();
    return 1;
  }

  CommonOptionsParser &OptionsParser = ExpectedParser.get();
  ClangTool Tool(OptionsParser.getCompilations(),
                 OptionsParser.getSourcePathList());

  return Tool.run(newFrontendActionFactory<CMLIRCFrontendAction>().get());
}
