#ifndef CMLIR_SUPPORT_PRINTING_H
#define CMLIR_SUPPORT_PRINTING_H

#include "llvm/Support/WithColor.h"

namespace mlir {

class Operation;

} // namespace mlir

namespace cmlir {

// Auto honors LLVM's --color option and terminal detection. Plain output is
// printed directly; highlighting preserves every byte of the printed MLIR.
auto printIR(mlir::Operation *, llvm::raw_ostream &,
             llvm::ColorMode = llvm::ColorMode::Auto) -> void;

} // namespace cmlir

#endif // CMLIR_SUPPORT_PRINTING_H
