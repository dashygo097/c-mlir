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
extern llvm::cl::opt<bool> funcInline;
extern llvm::cl::opt<bool> struct2Memref;
extern llvm::cl::opt<bool> raiseMemref2Affine;
extern llvm::cl::opt<bool> raiseSCF2Affine;
extern llvm::cl::opt<bool> fma;

extern llvm::cl::extrahelp commonHelp;

} // namespace cmlir::frontend::options

#endif // CLANG_TO_MLIR_ARGUMENT_LIST_H
