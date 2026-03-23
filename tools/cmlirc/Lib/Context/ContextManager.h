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

  [[nodiscard]] clang::ASTContext &ClangContext() noexcept { return *clangCtx; }
  [[nodiscard]] mlir::MLIRContext &MLIRContext() noexcept { return *mlirCtx; }
  [[nodiscard]] mlir::OpBuilder &Builder() noexcept { return *builder; }
  [[nodiscard]] mlir::ModuleOp &Module() noexcept { return *module; }

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
