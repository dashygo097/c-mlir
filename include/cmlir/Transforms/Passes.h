#ifndef CMLIR_PASSES_H
#define CMLIR_PASSES_H

#include "mlir/Pass/Pass.h"
#include <memory>

namespace cmlir {

// Common Optimization Passes
auto createMem2RegPass() -> std::unique_ptr<mlir::Pass>;
auto createConstPropPass() -> std::unique_ptr<mlir::Pass>;

// Loop Optimization Passes
auto createLoopUnrollPass() -> std::unique_ptr<mlir::Pass>;
auto createLoopVectorizePass() -> std::unique_ptr<mlir::Pass>;

// Conversion Passes (optional)
auto createStruct2MemrefPass() -> std::unique_ptr<mlir::Pass>;
auto createFMAPass() -> std::unique_ptr<mlir::Pass>;
auto createRaiseSCF2AffinePass() -> std::unique_ptr<mlir::Pass>;

void registerTransformPasses();

#define GEN_PASS_REGISTRATION
#include "cmlir/Transforms/Passes.h.inc"

} // namespace cmlir

#endif // CMLIR_PASSES_H
