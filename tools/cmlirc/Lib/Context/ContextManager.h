#ifndef CMLIRC_CONTEXT_MANAGER_H
#define CMLIRC_CONTEXT_MANAGER_H

#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/MLIRContext.h"
#include "clang/AST/ASTContext.h"

namespace cmlirc {

class ContextManager {
public:
  explicit ContextManager(clang::ASTContext *clangCtx);
  ~ContextManager() = default;

  [[nodiscard]] clang::ASTContext &ClangContext() noexcept {
    return *clang_context_;
  }
  [[nodiscard]] mlir::MLIRContext &MLIRContext() noexcept {
    return *mlir_context_;
  }
  [[nodiscard]] mlir::OpBuilder &Builder() noexcept { return *builder_; }
  [[nodiscard]] mlir::ModuleOp &Module() noexcept { return *module_; }

  void dump();
  void dump(llvm::raw_ostream &os);

private:
  clang::ASTContext *clang_context_;
  std::unique_ptr<mlir::MLIRContext> mlir_context_;
  std::unique_ptr<mlir::OpBuilder> builder_;
  std::unique_ptr<mlir::ModuleOp> module_;
};

} // namespace cmlirc

#endif // CMLIRC_CONTEXT_MANAGER_H
