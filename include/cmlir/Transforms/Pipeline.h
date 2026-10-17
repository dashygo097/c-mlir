#ifndef CMLIR_TRANSFORMS_PIPELINE_H
#define CMLIR_TRANSFORMS_PIPELINE_H

#include "mlir/IR/BuiltinOps.h"
#include "llvm/Support/Error.h"

namespace mlir {

class TimingScope;

} // namespace mlir

namespace cmlir {

struct PipelineOptions {
  bool optimize = true;
  bool dumpStages = false;
  bool inlineFunctions = false;
  bool promoteStructs = false;
  bool raiseSCFToAffine = false;
  bool raiseMemRefToAffine = false;
  // Explicitly permit floating-point contraction, without reassociation.
  bool upliftToFMA = false;
  // Optional caller-owned scope; it must outlive runPipeline.
  mlir::TimingScope *timingScope = nullptr;
};

// Always verifies, including when optimization is disabled. A failed pipeline
// may have already modified the module; use a clone when rollback is required.
auto runPipeline(mlir::ModuleOp module, const PipelineOptions &options = {})
    -> llvm::Error;

} // namespace cmlir

#endif // CMLIR_TRANSFORMS_PIPELINE_H
