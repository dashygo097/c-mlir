#ifndef CMLIRC_CONTEXT_MANAGER_H
#define CMLIRC_CONTEXT_MANAGER_H

#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/MLIRContext.h"
#include "clang/AST/ASTContext.h"

namespace cmlirc {

class ContextManager {
public:
  explicit ContextManager(clang::ASTContext *clangContext,
                          mlir::DialectRegistry *registry = nullptr);
  ~ContextManager() = default;

  [[nodiscard]] auto ClangContext() noexcept -> clang::ASTContext & { return *clangCtx; }
  [[nodiscard]] auto MLIRContext() noexcept -> mlir::MLIRContext & { return *mlirCtx; }
  [[nodiscard]] auto Builder() noexcept -> mlir::OpBuilder & { return *builder; }
  [[nodiscard]] auto Module() noexcept -> mlir::ModuleOp & { return *module; }

  void dump();
  void dump(llvm::raw_ostream &os);

private:
  clang::ASTContext *clangCtx;
  std::unique_ptr<mlir::MLIRContext> mlirCtx;
  std::unique_ptr<mlir::OpBuilder> builder;
  std::unique_ptr<mlir::ModuleOp> module;
};

} // namespace cmlirc

#endif // CMLIRC_CONTEXT_MANAGER_H
