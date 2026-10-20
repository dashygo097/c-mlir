#include "cmlir/Dialect.h"
#include "cmlir/Pipelines/Pipelines.h"
#include "cmlir/Transforms/Passes.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllExtensions.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"

auto main(int argc, char **argv) -> int {
  mlir::registerAllPasses();
  cmlir::registerTransformsPasses();
  cmlir::registerPipelines();

  mlir::DialectRegistry registry;
  registry.insert<cmlir::CMLIRDialect>();
  mlir::registerAllDialects(registry);
  mlir::registerAllExtensions(registry);
  return mlir::asMainReturnCode(
      mlir::MlirOptMain(argc, argv, "CMLIR optimizer driver\n", registry));
}
