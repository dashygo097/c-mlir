#ifndef CMLIRC_ARGUMENTLIST_H
#define CMLIRC_ARGUMENTLIST_H

#include "llvm/Support/CommandLine.h"

namespace cmlirc::options {
extern llvm::cl::OptionCategory toolOptions;

extern llvm::cl::opt<bool> Verbose;
extern llvm::cl::opt<std::string> FunctionName;
extern llvm::cl::opt<bool> MergeConstants;

extern llvm::cl::extrahelp CommonHelp;
} // namespace cmlirc::options

#endif
