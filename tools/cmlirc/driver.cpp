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
llvm::cl::opt<bool> Verbose("v", llvm::cl::init(false),
                            llvm::cl::desc("Enable verbose"),
                            llvm::cl::cat(toolOptions));

llvm::cl::opt<std::string>
    SystemRoot("sysroot", llvm::cl::init(""),
               llvm::cl::desc("Set the system root path"),
               llvm::cl::value_desc("path"), llvm::cl::cat(toolOptions));

llvm::cl::opt<std::string> OutputFile("o",
                                      llvm::cl::desc("Write output to <file>"),
                                      llvm::cl::value_desc("file"),
                                      llvm::cl::init("-"));

llvm::cl::opt<std::string>
    FunctionName("function", llvm::cl::init(""),
                 llvm::cl::desc("Name of the function to compile"),
                 llvm::cl::cat(toolOptions));

// Passes
llvm::cl::opt<bool>
    Struct2Memref("struct-to-memref", llvm::cl::init(false),
                  llvm::cl::desc("Enable struct to memref promotion"),
                  llvm::cl::cat(toolOptions));

llvm::cl::opt<bool> RaiseSCF2Affine("raise-scf-to-affine",
                                    llvm::cl::init(false),
                                    llvm::cl::desc("Raise scf to affine"),
                                    llvm::cl::cat(toolOptions));

llvm::cl::opt<bool> FMA("fma", llvm::cl::init(false),
                        llvm::cl::desc("Enable fused multiply-add (FMA)"),
                        llvm::cl::cat(toolOptions));

llvm::cl::extrahelp CommonHelp(CommonOptionsParser::HelpMessage);

} // namespace cmlirc::options

int main(int argc, const char **argv) {
  auto ExpectedParser =
      CommonOptionsParser::create(argc, argv, options::toolOptions);

  if (!ExpectedParser) {
    llvm::WithColor::error()
        << "cmlirc: " << toString(ExpectedParser.takeError()) << "\n";
    return 1;
  }

  CommonOptionsParser &OptionsParser = ExpectedParser.get();
  ClangTool Tool(OptionsParser.getCompilations(),
                 OptionsParser.getSourcePathList());
  Tool.appendArgumentsAdjuster(getInsertArgumentAdjuster(
      {"-isysroot", options::SystemRoot}, ArgumentInsertPosition::BEGIN));

  llvm::raw_ostream *out;
  std::unique_ptr<llvm::raw_fd_ostream> fileOut;

  if (options::OutputFile == "-") {
    out = &llvm::outs();
  } else {
    std::error_code ec;
    fileOut = std::make_unique<llvm::raw_fd_ostream>(options::OutputFile, ec,
                                                     llvm::sys::fs::OF_None);

    if (ec) {
      llvm::WithColor::error() << "cmlirc: cannot open '" << options::OutputFile
                               << "': " << ec.message() << "\n";
      return 1;
    }
    out = fileOut.get();
  }

  auto factory = std::make_unique<CMLIRActionFactory>(out);
  return Tool.run(factory.get());
}
