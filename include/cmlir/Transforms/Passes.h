#ifndef CMLIR_PASSES_H
#define CMLIR_PASSES_H

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Pass/Pass.h"
#include <memory>

namespace cmlir {

std::unique_ptr<mlir::OperationPass<mlir::func::FuncOp>> createMem2RegPass();
std::unique_ptr<mlir::OperationPass<mlir::func::FuncOp>> createConstPropPass();

void registerTransformPasses();

#define GEN_PASS_REGISTRATION
#include "cmlir/Transforms/Passes.h.inc"

} // namespace cmlir

#endif // CMLIR_PASSES_H
