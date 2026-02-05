#include "Lib/FrontendAction.h"
#include "clang/Frontend/FrontendActions.h"
#include "clang/Tooling/CommonOptionsParser.h"
#include "clang/Tooling/Tooling.h"
#include "llvm/Support/CommandLine.h"

using namespace clang;
using namespace clang::tooling;
using namespace cmlirc;

static llvm::cl::OptionCategory toolOptions("CMLIRC Options");

static llvm::cl::opt<bool> ShowAST("dump-ast", llvm::cl::init(false),
                                   llvm::cl::desc("Dump AST"),
                                   llvm::cl::cat(toolOptions));

static llvm::cl::extrahelp CommonHelp(CommonOptionsParser::HelpMessage);

int main(int argc, const char **argv) {
  auto ExpectedParser = CommonOptionsParser::create(argc, argv, toolOptions);

  if (!ExpectedParser) {
    llvm::errs() << ExpectedParser.takeError();
    return 1;
  }

  CommonOptionsParser &OptionsParser = ExpectedParser.get();
  ClangTool Tool(OptionsParser.getCompilations(),
                 OptionsParser.getSourcePathList());

  if (ShowAST) {
    return Tool.run(newFrontendActionFactory<CMLIRCFrontendAction>().get());
  }

  return Tool.run(newFrontendActionFactory<clang::SyntaxOnlyAction>().get());
}
