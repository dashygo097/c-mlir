#include "cmlir/Dialect.h"
#include "cmlir/CMLIROpsDialect.cpp.inc"
#include "cmlir/Ops.h"

#define GET_OP_CLASSES
#include "cmlir/CMLIROps.cpp.inc"
#undef GET_OP_CLASSES

namespace cmlir {

auto CMLIRDialect::initialize() -> void {
  addOperations<
#define GET_OP_LIST
#include "cmlir/CMLIROps.cpp.inc"
      >();
}

} // namespace cmlir
