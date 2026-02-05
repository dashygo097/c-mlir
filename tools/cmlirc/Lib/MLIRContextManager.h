#ifndef CMLIRC_MLIR_CONTEXT_H
#define CMLIRC_MLIR_CONTEXT_H

#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/MLIRContext.h"

namespace cmlirc {

class MLIRContextManager {
public:
  explicit MLIRContextManager();
  ~MLIRContextManager() = default;

  [[nodiscard]] mlir::MLIRContext &Context() noexcept { return *context_; }
  [[nodiscard]] mlir::OpBuilder &Builder() noexcept { return *builder_; }
  [[nodiscard]] mlir::ModuleOp &Module() noexcept { return *module_; }

  void dump();

private:
  std::unique_ptr<mlir::MLIRContext> context_;
  std::unique_ptr<mlir::OpBuilder> builder_;
  std::unique_ptr<mlir::ModuleOp> module_;
};

} // namespace cmlirc

#endif // CMLIRC_MLIR_CONTEXT_H
