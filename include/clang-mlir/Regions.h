#ifndef CLANG_MLIR_REGIONS_H
#define CLANG_MLIR_REGIONS_H

#include "mlir/IR/Builders.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/STLFunctionalExtras.h"

namespace clang_mlir {

auto hasTerminator(mlir::Block *block) -> bool;
// Restores the insertion point on every exit and propagates callback failure.
// A null continuation produces scf.yield; otherwise it produces cf.br.
auto emitArm(mlir::OpBuilder &builder, mlir::Location loc, mlir::Block &arm,
             mlir::Block *continuation,
             llvm::function_ref<mlir::LogicalResult()> emitBody)
    -> mlir::LogicalResult;

} // namespace clang_mlir

#endif // CLANG_MLIR_REGIONS_H
