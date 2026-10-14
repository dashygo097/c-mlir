#ifndef CMLIR_FRONTEND_POINTER_ARITHMETIC_H
#define CMLIR_FRONTEND_POINTER_ARITHMETIC_H

#include "mlir/IR/Builders.h"

namespace cmlir::frontend {

// The offset counts pointee objects, not bytes. Preserve memref descriptors
// when available; the storage element type supplies scaling for raw pointers.
auto emitPointerOffset(mlir::OpBuilder &, mlir::Location, mlir::Value pointer,
                       mlir::Value offset, mlir::Type storageElement,
                       bool signedOffset, bool subtract) -> mlir::Value;

} // namespace cmlir::frontend

#endif // CMLIR_FRONTEND_POINTER_ARITHMETIC_H
