#ifndef CMLIRC_ARGUMENTLIST_H
#define CMLIRC_ARGUMENTLIST_H

#include "llvm/Support/CommandLine.h"

namespace cmlirc::options {
extern llvm::cl::OptionCategory toolOptions;

extern llvm::cl::opt<bool> Verbose;
extern llvm::cl::opt<std::string> FunctionName;
extern llvm::cl::opt<std::string> OutputFile;

extern llvm::cl::opt<bool> ConstProp;
extern llvm::cl::opt<bool> FuncInline;
extern llvm::cl::opt<bool> SSCP;
extern llvm::cl::opt<bool> Mem2Reg;
extern llvm::cl::opt<bool> Canonicalize;
extern llvm::cl::opt<bool> CSE;
extern llvm::cl::opt<bool> LICM;
extern llvm::cl::opt<bool> SymbolDCE;

extern llvm::cl::extrahelp CommonHelp;
} // namespace cmlirc::options

#endif
