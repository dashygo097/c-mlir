#ifndef CMLIR_PIPELINES_PIPELINES_H
#define CMLIR_PIPELINES_PIPELINES_H

#include "mlir/IR/BuiltinOps.h"
#include "llvm/Support/Error.h"

namespace mlir {

class DialectRegistry;
class OpPassManager;
class TimingScope;

} // namespace mlir

namespace cmlir {

struct PipelineOptions {
  bool optimize = true;
  bool dumpStages = false;
  bool inlineFunctions = false;
  bool promoteStructs = false;
  // Recover structured control flow after SSA promotion and optional inlining.
  bool liftCFToSCF = true;
  bool raiseSCFToAffine = false;
  bool raiseMemRefToAffine = false;
  // Fuse operations only when their existing flags permit contraction.
  bool upliftToFMA = false;
  // Optional caller-owned scope; it must outlive runPipeline.
  mlir::TimingScope *timingScope = nullptr;
};

// Register the external models needed by the shared pipeline.
auto registerPipelineInterfaces(mlir::DialectRegistry &registry) -> void;

// Append the default pass sequence without executing it. Execution controls
// (optimize, dumpStages and timingScope) belong to runPipeline only.
auto buildDefaultPipeline(mlir::OpPassManager &pm,
                          const PipelineOptions &options = {}) -> void;

// Register default-pipeline and its transformation options.
auto registerPipelines() -> void;

// Always verifies, including when optimization is disabled. A failed pipeline
// may have already modified the module; use a clone when rollback is required.
auto runPipeline(mlir::ModuleOp module, const PipelineOptions &options = {})
    -> llvm::Error;

} // namespace cmlir

#endif // CMLIR_PIPELINES_PIPELINES_H
