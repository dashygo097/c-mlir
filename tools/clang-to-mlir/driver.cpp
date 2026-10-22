#ifndef DEFAULT_SYSROOT
#define DEFAULT_SYSROOT ""
#endif

#include "./ArgumentList.h"
#include "FrontendAction.h"
#include "mlir/Support/Timing.h"
#include "clang/Tooling/CommonOptionsParser.h"
#include "clang/Tooling/Tooling.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/WithColor.h"

using namespace clang;
using namespace clang::tooling;
using namespace cmlir::frontend;

namespace cmlir::frontend::options {

llvm::cl::OptionCategory toolOptions("Tool Options");

// General options
llvm::cl::opt<bool> verbose("v", llvm::cl::init(false),
                            llvm::cl::desc("Enable verbose"),
                            llvm::cl::cat(toolOptions));

llvm::cl::opt<std::string>
    systemRoot("sysroot", llvm::cl::init(DEFAULT_SYSROOT),
               llvm::cl::desc("Set the system root path"),
               llvm::cl::value_desc("path"), llvm::cl::cat(toolOptions));

llvm::cl::opt<std::string> outputFile("o",
                                      llvm::cl::desc("Write output to <file>"),
                                      llvm::cl::value_desc("file"),
                                      llvm::cl::init("-"),
                                      llvm::cl::cat(toolOptions));

llvm::cl::opt<std::string>
    functionName("function", llvm::cl::init(""),
                 llvm::cl::desc("Name of the function to compile"),
                 llvm::cl::cat(toolOptions));

llvm::cl::opt<bool>
    dumpStages("dump-stages", llvm::cl::init(false),
               llvm::cl::desc("Print verified compilation stages to stderr"),
               llvm::cl::cat(toolOptions));

llvm::cl::opt<bool>
    pipelineTiming("pipeline-timing", llvm::cl::init(false),
                   llvm::cl::desc("Print pass and analysis timings"),
                   llvm::cl::cat(toolOptions));

// Passes
llvm::cl::opt<bool> disableOpt("disable-opt", llvm::cl::init(false),
                               llvm::cl::desc("Disable optimizations"),
                               llvm::cl::cat(toolOptions));

llvm::cl::opt<bool> inlinePass("inline", llvm::cl::init(false),
                               llvm::cl::desc("Enable function inlining"),
                               llvm::cl::cat(toolOptions));

llvm::cl::opt<bool>
    structToMemref("struct-to-memref", llvm::cl::init(false),
                   llvm::cl::desc("Enable struct to memref promotion"),
                   llvm::cl::cat(toolOptions));

llvm::cl::opt<bool>
    liftCFToSCF("lift-cf-to-scf", llvm::cl::init(true),
                llvm::cl::desc("Lift control flow to SCF (enabled by default)"),
                llvm::cl::cat(toolOptions));

llvm::cl::opt<bool>
    affineRaiseFromMemref("affine-raise-from-memref", llvm::cl::init(false),
                          llvm::cl::desc("Raise memref to affine"),
                          llvm::cl::cat(toolOptions));

llvm::cl::opt<bool> raiseSCFToAffine("raise-scf-to-affine",
                                     llvm::cl::init(false),
                                     llvm::cl::desc("Raise scf to affine"),
                                     llvm::cl::cat(toolOptions));

llvm::cl::opt<bool> mathUpliftToFMA(
    "math-uplift-to-fma", llvm::cl::init(false),
    llvm::cl::desc("Uplift permitted floating-point contractions to math.fma"),
    llvm::cl::cat(toolOptions));

llvm::cl::extrahelp commonHelp(CommonOptionsParser::HelpMessage);

} // namespace cmlir::frontend::options

auto main(int argc, const char **argv) -> int {
  auto expectedParser =
      CommonOptionsParser::create(argc, argv, options::toolOptions);

  if (!expectedParser) {
    llvm::WithColor::error()
        << "clang-to-mlir: " << toString(expectedParser.takeError()) << "\n";
    return 1;
  }

  CommonOptionsParser &optionsParser = expectedParser.get();
  ClangTool tool(optionsParser.getCompilations(),
                 optionsParser.getSourcePathList());
  if (!options::systemRoot.empty()) {
    tool.appendArgumentsAdjuster(getInsertArgumentAdjuster(
        {"-isysroot", options::systemRoot}, ArgumentInsertPosition::BEGIN));
  }

  llvm::raw_ostream *out;
  std::unique_ptr<llvm::raw_fd_ostream> fileOut;

  if (options::outputFile == "-") {
    out = &llvm::outs();
  } else {
    std::error_code ec;
    fileOut = std::make_unique<llvm::raw_fd_ostream>(options::outputFile, ec,
                                                     llvm::sys::fs::OF_None);

    if (ec) {
      llvm::WithColor::error()
          << "clang-to-mlir: cannot open '" << options::outputFile
          << "': " << ec.message() << "\n";
      return 1;
    }
    out = fileOut.get();
  }

  cmlir::frontend::CodeGenOptions codegen;
  codegen.function = options::functionName;
  codegen.traceLowering = options::verbose;
  cmlir::PipelineOptions pipeline;
  pipeline.optimize = !options::disableOpt;
  pipeline.dumpStages = options::dumpStages;
  pipeline.inlineFunctions = options::inlinePass;
  pipeline.promoteStructs = options::structToMemref;
  pipeline.liftCFToSCF = options::liftCFToSCF;
  pipeline.raiseMemRefToAffine = options::affineRaiseFromMemref;
  pipeline.raiseSCFToAffine = options::raiseSCFToAffine;
  pipeline.upliftToFMA = options::mathUpliftToFMA;
  mlir::DefaultTimingManager timing;
  timing.setEnabled(options::pipelineTiming);
  timing.setDisplayMode(mlir::DefaultTimingManager::DisplayMode::List);
  auto timingScope = timing.getRootScope();
  if (options::pipelineTiming) {
    pipeline.timingScope = &timingScope;
  }
  auto factory = std::make_unique<ActionFactory>(*out, codegen, pipeline);
  return tool.run(factory.get());
}
