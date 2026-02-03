#include "cmlir/Dialect.h"
#include "cmlir/Ops.h"
#include "mlir/IR/DialectImplementation.h"

using namespace mlir;
using namespace cmlir;

void CMLIRDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "cmlir/CMLIROps.cpp.inc"
      >();
}
