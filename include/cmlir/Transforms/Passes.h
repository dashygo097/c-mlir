#ifndef CMLIR_PASSES_H
#define CMLIR_PASSES_H

#include "mlir/Pass/Pass.h"
#include <memory>

namespace cmlir {

// Common optimization passes
auto createArithFoldCastsPass() -> std::unique_ptr<mlir::Pass>;

// Explicit floating-point policy; permits contraction without reassociation.
auto createEnableFPContractPass() -> std::unique_ptr<mlir::Pass>;

// Loop optimization passes.
auto createLoopUnrollPass() -> std::unique_ptr<mlir::Pass>;
auto createLoopVectorizePass() -> std::unique_ptr<mlir::Pass>;

// Representation conversion passes (optional).
auto createStruct2MemrefPass() -> std::unique_ptr<mlir::Pass>;

#define GEN_PASS_REGISTRATION
#include "cmlir/Transforms/Passes.h.inc"

} // namespace cmlir

#endif // CMLIR_PASSES_H
