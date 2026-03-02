#ifndef CMLIR_PASSES_H
#define CMLIR_PASSES_H

#include "mlir/Pass/Pass.h"
#include <memory>

namespace cmlir {

// Common Optimization Passes
std::unique_ptr<mlir::Pass> createMem2RegPass();
std::unique_ptr<mlir::Pass> createConstPropPass();

// Loop Optimization Passes
std::unique_ptr<mlir::Pass> createLoopUnrollPass();
std::unique_ptr<mlir::Pass> createLoopVectorizePass();

// Conversion Passes (optional)
std::unique_ptr<mlir::Pass> createStruct2MemrefPass();
std::unique_ptr<mlir::Pass> createFMAPass();
std::unique_ptr<mlir::Pass> createRaiseSCF2AffinePass();

void registerTransformPasses();

#define GEN_PASS_REGISTRATION
#include "cmlir/Transforms/Passes.h.inc"

} // namespace cmlir

#endif // CMLIR_PASSES_H
