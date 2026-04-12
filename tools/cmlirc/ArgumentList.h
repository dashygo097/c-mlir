#ifndef CMLIRC_ARGUMENTLIST_H
#define CMLIRC_ARGUMENTLIST_H

#include "llvm/Support/CommandLine.h"

namespace cmlirc::options {
extern llvm::cl::OptionCategory toolOptions;

extern llvm::cl::opt<bool> verbose;
extern llvm::cl::opt<std::string> systemRoot;
extern llvm::cl::opt<std::string> outputFile;
extern llvm::cl::opt<std::string> functionName;

extern llvm::cl::opt<bool> disableOpt;
extern llvm::cl::opt<bool> arithCastProp;
extern llvm::cl::opt<bool> funcInline;
extern llvm::cl::opt<bool> struct2Memref;
extern llvm::cl::opt<bool> raiseSCF2Affine;
extern llvm::cl::opt<bool> fma;

extern llvm::cl::extrahelp commonHelp;
} // namespace cmlirc::options

#endif
