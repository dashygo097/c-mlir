#ifndef CHWC_UTILS_BUILDER_H
#define CHWC_UTILS_BUILDER_H

#include "mlir/IR/Builders.h"
#include "mlir/IR/Operation.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"

namespace chwc::utils {

inline auto createOneResultOp(mlir::OpBuilder &builder, mlir::Location loc,
                              llvm::StringRef opName,
                              llvm::ArrayRef<mlir::Value> operands,
                              mlir::Type resultType) -> mlir::Value {
  mlir::OperationState state(loc, opName);
  state.addOperands(operands);
  state.addTypes(resultType);

  mlir::Operation *op = builder.create(state);
  if (!op || op->getNumResults() == 0) {
    return nullptr;
  }

  return op->getResult(0);
}

} // namespace chwc::utils

#endif // CHWC_UTILS_BUILDER_H
