#ifndef CLANG_TO_MLIR_ARGUMENT_LIST_H
#define CLANG_TO_MLIR_ARGUMENT_LIST_H

#include "llvm/Support/CommandLine.h"

namespace cmlir::frontend::options {

extern llvm::cl::OptionCategory toolOptions;

extern llvm::cl::opt<bool> verbose;
extern llvm::cl::opt<std::string> systemRoot;
extern llvm::cl::opt<std::string> outputFile;
extern llvm::cl::opt<std::string> functionName;

extern llvm::cl::opt<bool> disableOpt;
extern llvm::cl::opt<bool> dumpStages;
extern llvm::cl::opt<bool> inlinePass;
extern llvm::cl::opt<bool> structToMemref;
extern llvm::cl::opt<bool> affineRaiseFromMemref;
extern llvm::cl::opt<bool> raiseSCFToAffine;
extern llvm::cl::opt<bool> mathUpliftToFMA;

extern llvm::cl::extrahelp commonHelp;

} // namespace cmlir::frontend::options

#endif // CLANG_TO_MLIR_ARGUMENT_LIST_H
