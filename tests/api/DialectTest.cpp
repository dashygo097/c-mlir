#include "cmlir/Dialect.h"
#include "cmlir/Ops.h"
#include "mlir/IR/MLIRContext.h"
#include "llvm/Support/raw_ostream.h"

auto main() -> int {
  mlir::MLIRContext ir;
  auto *dialect = ir.getOrLoadDialect<cmlir::CMLIRDialect>();
  if (!dialect || dialect->getNamespace() != "cmlir") {
    return 1;
  }
  llvm::outs() << "CMLIRDialect API checks passed\n";
  return 0;
}
