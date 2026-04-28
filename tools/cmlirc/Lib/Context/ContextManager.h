#ifndef CMLIRC_CONTEXT_MANAGER_H
#define CMLIRC_CONTEXT_MANAGER_H

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"
#include "clang/AST/ASTContext.h"
#include <memory>

namespace cmlirc {

class CMLIRContextManager {
public:
  explicit CMLIRContextManager(clang::ASTContext *clangContext,
                               mlir::DialectRegistry *registry = nullptr);
  ~CMLIRContextManager() = default;

  [[nodiscard]] auto ClangContext() noexcept -> clang::ASTContext & {
    return *clangCtx;
  }
  [[nodiscard]] auto MLIRContext() noexcept -> mlir::MLIRContext & {
    return *mlirCtx;
  }
  [[nodiscard]] auto Builder() noexcept -> mlir::OpBuilder & {
    return *builder;
  }
  [[nodiscard]] auto Module() noexcept -> mlir::ModuleOp {
    return module.get();
  }

  void dump();
  void dump(llvm::raw_ostream &os);

private:
  clang::ASTContext *clangCtx;
  std::unique_ptr<mlir::MLIRContext> mlirCtx;
  std::unique_ptr<mlir::OpBuilder> builder;
  mlir::OwningOpRef<mlir::ModuleOp> module;
};

} // namespace cmlirc

#endif // CMLIRC_CONTEXT_MANAGER_H
