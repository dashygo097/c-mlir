#ifndef CMLIR_PASSES_H
#define CMLIR_PASSES_H

#include "mlir/Pass/Pass.h"
#include <memory>

namespace cmlir {

std::unique_ptr<mlir::Pass> createMem2RegPass();

void registerTransformPasses();

#define GEN_PASS_REGISTRATION
#include "cmlir/Transforms/Passes.h.inc"

} // namespace cmlir

#endif // CMLIR_PASSES_H
