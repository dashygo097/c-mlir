#ifndef CLANG_MLIR_MEMREFABI_H
#define CLANG_MLIR_MEMREFABI_H

#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/WithColor.h"

namespace clang_mlir {

auto memrefSameElementType(mlir::MemRefType src, mlir::MemRefType dst) -> bool;
auto memrefIsScalarAddressToPointer(mlir::MemRefType src, mlir::MemRefType dst)
    -> bool;
auto reinterpretScalarMemRefAsPointerMemRef(mlir::OpBuilder &builder,
                                            mlir::Location loc,
                                            mlir::Value value,
                                            mlir::MemRefType dstType)
    -> mlir::Value;
auto coerceMemRef(mlir::OpBuilder &builder, mlir::Location loc,
                  mlir::Value value, mlir::MemRefType dstType) -> mlir::Value;

} // namespace clang_mlir

#endif // CLANG_MLIR_MEMREFABI_H
