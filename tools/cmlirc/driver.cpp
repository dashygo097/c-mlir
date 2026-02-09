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

llvm::cl::opt<std::string>
    FunctionName("function", llvm::cl::init(""),
                 llvm::cl::desc("Name of the function to compile"),
                 llvm::cl::cat(toolOptions));

llvm::cl::opt<bool> FuncInline("func-inline", llvm::cl::init(false),
                               llvm::cl::desc("Enable function inlining"),
                               llvm::cl::cat(toolOptions));

llvm::cl::opt<bool>
    SSCP("sscp", llvm::cl::init(false),
         llvm::cl::desc("Enable sparse simple constant propagation (SSCP)"),
         llvm::cl::cat(toolOptions));

llvm::cl::opt<bool> Canonicalize("canonicalize", llvm::cl::init(false),
                                 llvm::cl::desc("Enable canonicalization"),
                                 llvm::cl::cat(toolOptions));

llvm::cl::opt<bool>
    CSE("cse", llvm::cl::init(false),
        llvm::cl::desc("Enable common subexpression elimination (CSE)"),
        llvm::cl::cat(toolOptions));

llvm::cl::opt<bool>
    LICM("licm", llvm::cl::init(false),
         llvm::cl::desc("Enable loop-invariant code motion (LICM)"),
         llvm::cl::cat(toolOptions));

llvm::cl::opt<bool>
    SymbolDCE("symdce", llvm::cl::init(false),
              llvm::cl::desc("Enable symbol dead code elimination"),
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
