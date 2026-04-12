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

llvm::cl::opt<std::string>
    functionName("function", llvm::cl::init(""),
                 llvm::cl::desc("Name of the function to compile"),
                 llvm::cl::cat(toolOptions));

// Passes
llvm::cl::opt<bool> disableOpt("disable-opt", llvm::cl::init(false),
                               llvm::cl::desc("Disable optimizations"),
                               llvm::cl::cat(toolOptions));

llvm::cl::opt<bool> funcInline("func-inline", llvm::cl::init(false),
                               llvm::cl::desc("Enable function inlining"),
                               llvm::cl::cat(toolOptions));

llvm::cl::opt<bool>
    arithCastProp("arith-cast-prop", llvm::cl::init(false),
                  llvm::cl::desc("Enable Arith cast propagations"),
                  llvm::cl::cat(toolOptions));

llvm::cl::opt<bool>
    struct2Memref("struct-to-memref", llvm::cl::init(false),
                  llvm::cl::desc("Enable struct to memref promotion"),
                  llvm::cl::cat(toolOptions));

llvm::cl::opt<bool> raiseSCF2Affine("raise-scf-to-affine",
                                    llvm::cl::init(false),
                                    llvm::cl::desc("Raise scf to affine"),
                                    llvm::cl::cat(toolOptions));

llvm::cl::opt<bool> fma("fma", llvm::cl::init(false),
                        llvm::cl::desc("Enable fused multiply-add (FMA)"),
                        llvm::cl::cat(toolOptions));

llvm::cl::extrahelp commonHelp(CommonOptionsParser::HelpMessage);

} // namespace cmlirc::options

int main(int argc, const char **argv) {
  auto expectedParser =
      CommonOptionsParser::create(argc, argv, options::toolOptions);

  if (!expectedParser) {
    llvm::WithColor::error()
        << "cmlirc: " << toString(expectedParser.takeError()) << "\n";
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
      llvm::WithColor::error() << "cmlirc: cannot open '" << options::outputFile
                               << "': " << ec.message() << "\n";
      return 1;
    }
    out = fileOut.get();
  }

  auto factory = std::make_unique<CMLIRActionFactory>(out);
  return tool.run(factory.get());
}
