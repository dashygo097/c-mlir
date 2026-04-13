#include "./ArgumentList.h"
#include "clang/Frontend/FrontendActions.h"
#include "clang/Tooling/CommonOptionsParser.h"
#include "clang/Tooling/Tooling.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/WithColor.h"

using namespace clang;
using namespace clang::tooling;
using namespace chwc;

namespace chwc::options {
llvm::cl::OptionCategory toolOptions("CHWC Options");

// General options
llvm::cl::opt<bool> verbose("v", llvm::cl::init(false),
                            llvm::cl::desc("Enable verbose"),
                            llvm::cl::cat(toolOptions));

llvm::cl::opt<std::string>
    systemRoot("sysroot", llvm::cl::init(""),
               llvm::cl::desc("Set the system root path"),
               llvm::cl::value_desc("path"), llvm::cl::cat(toolOptions));

llvm::cl::opt<std::string> outputFile("o",
                                      llvm::cl::desc("Write output to <file>"),
                                      llvm::cl::value_desc("file"),
                                      llvm::cl::init("-"));

// Passes
llvm::cl::opt<bool> disableOpt("disable-opt", llvm::cl::init(false),
                               llvm::cl::desc("Disable optimizations"),
                               llvm::cl::cat(toolOptions));

llvm::cl::extrahelp commonHelp(CommonOptionsParser::HelpMessage);

} // namespace chwc::options

int main(int argc, const char **argv) {
  auto expectedParser =
      CommonOptionsParser::create(argc, argv, options::toolOptions);

  if (!expectedParser) {
    llvm::WithColor::error()
        << "chwc: " << toString(expectedParser.takeError()) << "\n";
    return 1;
  }

  CommonOptionsParser &optionsParser = expectedParser.get();
  ClangTool tool(optionsParser.getCompilations(),
                 optionsParser.getSourcePathList());
  tool.appendArgumentsAdjuster(getInsertArgumentAdjuster(
      {"-isysroot", options::systemRoot}, ArgumentInsertPosition::BEGIN));

  llvm::raw_ostream *out;
  std::unique_ptr<llvm::raw_fd_ostream> fileOut;

  if (options::outputFile == "-") {
    out = &llvm::outs();
  } else {
    std::error_code ec;
    fileOut = std::make_unique<llvm::raw_fd_ostream>(options::outputFile, ec,
                                                     llvm::sys::fs::OF_None);

    if (ec) {
      llvm::WithColor::error() << "chwc: cannot open '" << options::outputFile
                               << "': " << ec.message() << "\n";
      return 1;
    }
    out = fileOut.get();
  }

  return tool.run(newFrontendActionFactory<clang::SyntaxOnlyAction>().get());
}
