#ifndef CLANG_MLIR_CONTEXT_H
#define CLANG_MLIR_CONTEXT_H

#include "clang-mlir/Source.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"

namespace clang_mlir {

// Backend-neutral borrowed session. Dialect registration and module ownership
// belong to the caller; constructing this view never loads backend dialects.
class Context {
public:
  Context(clang::ASTContext &ast, mlir::ModuleOp module)
      : astContext(ast), irContext(*module.getContext()), opBuilder(&irContext),
        moduleOp(module), sourceInfo(ast, irContext) {}

  Context(const Context &) = delete;
  auto operator=(const Context &) -> Context & = delete;

  auto ast() -> clang::ASTContext & { return astContext; }

  auto ir() -> mlir::MLIRContext & { return irContext; }

  auto builder() -> mlir::OpBuilder & { return opBuilder; }

  auto module() -> mlir::ModuleOp { return moduleOp; }

  auto source() -> Source & { return sourceInfo; }

private:
  clang::ASTContext &astContext;
  mlir::MLIRContext &irContext;
  mlir::OpBuilder opBuilder;
  mlir::ModuleOp moduleOp;
  Source sourceInfo;
};

} // namespace clang_mlir

#endif // CLANG_MLIR_CONTEXT_H
