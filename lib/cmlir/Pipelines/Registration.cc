#include "cmlir/Pipelines/Pipelines.h"
#include "mlir/Pass/PassOptions.h"
#include "mlir/Pass/PassRegistry.h"

namespace cmlir {

namespace {

struct DefaultPipelineOptions
    : mlir::PassPipelineOptions<DefaultPipelineOptions> {
  Option<bool> inlineFunctions{*this, "inline",
                               llvm::cl::desc("Enable function inlining"),
                               llvm::cl::init(false)};
  Option<bool> promoteStructs{
      *this, "struct-to-memref",
      llvm::cl::desc("Enable homogeneous struct promotion"),
      llvm::cl::init(false)};
  Option<bool> liftCFToSCF{*this, "lift-cf-to-scf",
                           llvm::cl::desc("Recover structured control flow"),
                           llvm::cl::init(true)};
  Option<bool> raiseSCFToAffine{*this, "raise-scf-to-affine",
                                llvm::cl::desc("Raise SCF loops to Affine"),
                                llvm::cl::init(false)};
  Option<bool> raiseMemRefToAffine{
      *this, "affine-raise-from-memref",
      llvm::cl::desc("Raise memory accesses to Affine"), llvm::cl::init(false)};
  Option<bool> upliftToFMA{
      *this, "math-uplift-to-fma",
      llvm::cl::desc(
          "Uplift permitted floating-point contractions to math.fma"),
      llvm::cl::init(false)};
};

} // namespace

auto registerPipelines() -> void {
  mlir::PassPipelineRegistration<DefaultPipelineOptions>(
      "default-pipeline", "Run the shared CMLIR optimization pipeline",
      [](mlir::OpPassManager &pm,
         const DefaultPipelineOptions &options) -> void {
        PipelineOptions pipeline;
        pipeline.inlineFunctions = options.inlineFunctions;
        pipeline.promoteStructs = options.promoteStructs;
        pipeline.liftCFToSCF = options.liftCFToSCF;
        pipeline.raiseSCFToAffine = options.raiseSCFToAffine;
        pipeline.raiseMemRefToAffine = options.raiseMemRefToAffine;
        pipeline.upliftToFMA = options.upliftToFMA;
        buildDefaultPipeline(pm, pipeline);
      });
}

} // namespace cmlir
