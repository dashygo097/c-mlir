#ifndef CHWC_ARGUMENTLIST_H
#define CHWC_ARGUMENTLIST_H

#include "llvm/Support/CommandLine.h"

namespace chwc::options {
extern llvm::cl::OptionCategory toolOptions;

extern llvm::cl::opt<bool> verbose;
extern llvm::cl::opt<std::string> systemRoot;
extern llvm::cl::opt<std::string> outputFile;
extern llvm::cl::opt<std::string> moduleName;

extern llvm::cl::opt<bool> disableOpt;

extern llvm::cl::extrahelp commonHelp;
} // namespace chwc::options

#endif
