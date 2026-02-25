#include "cmlir/Dialect.h"

namespace cmlir {
void CMLIRDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "cmlir/CMLIROps.cpp.inc"
      >();
}

} // namespace cmlir
