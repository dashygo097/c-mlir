#ifndef CMLIR_TRANSFORMS_CAST_FOLDING_H
#define CMLIR_TRANSFORMS_CAST_FOLDING_H

#include "mlir/Support/LogicalResult.h"

namespace mlir {
class Operation;
} // namespace mlir

namespace cmlir {

auto foldExactIndexRoundTrips(mlir::Operation *) -> mlir::LogicalResult;

} // namespace cmlir

#endif // CMLIR_TRANSFORMS_CAST_FOLDING_H
