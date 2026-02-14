#include "./ArgumentList.h"
#include "./Lib/ActionFactory.h"
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

llvm::cl::opt<std::string> OutputFile("o",
                                      llvm::cl::desc("Write output to <file>"),
                                      llvm::cl::value_desc("file"),
                                      llvm::cl::init("-"));

llvm::cl::opt<bool> FuncInline("func-inline", llvm::cl::init(false),
                               llvm::cl::desc("Enable function inlining"),
                               llvm::cl::cat(toolOptions));

llvm::cl::opt<bool>
    SSCP("sscp", llvm::cl::init(false),
         llvm::cl::desc("Enable sparse simple constant propagation (SSCP)"),
         llvm::cl::cat(toolOptions));

llvm::cl::opt<bool>
    Mem2Reg("mem2reg", llvm::cl::init(false),
            llvm::cl::desc("Enable memory to register promotion (mem2reg)"),
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

  llvm::raw_ostream *out;
  std::unique_ptr<llvm::raw_fd_ostream> fileOut;

  if (options::OutputFile == "-") {
    out = &llvm::outs();
  } else {
    std::error_code ec;
    fileOut = std::make_unique<llvm::raw_fd_ostream>(options::OutputFile, ec,
                                                     llvm::sys::fs::OF_None);

    if (ec) {
      llvm::errs() << "Error: cannot open '" << options::OutputFile
                   << "': " << ec.message() << "\n";
      return 1;
    }
    out = fileOut.get();
  }

  auto factory = std::make_unique<CMLIRActionFactory>(out);
  return Tool.run(factory.get());
}
