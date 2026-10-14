#ifndef CMLIR_FRONTEND_H
#define CMLIR_FRONTEND_H

#include "cmlir/Frontend/LoopHints.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/OwningOpRef.h"
#include "llvm/Support/Error.h"
#include <string>

namespace clang {

class ASTContext;

} // namespace clang

namespace mlir {

class DialectRegistry;
class MLIRContext;

} // namespace mlir

namespace cmlir::frontend {

struct CodeGenOptions {
  std::string function;
  bool traceLowering = false;
};

// Call before constructing a context, or append the registry to an existing
// one.
auto registerDialects(mlir::DialectRegistry &registry) -> void;

// Borrows the AST only during this call. The caller must keep the MLIR context
// alive for as long as the returned module is alive. Failed lowering returns no
// partial module and reports source diagnostics through Clang's diagnostics.
auto generateModule(clang::ASTContext &ast, mlir::MLIRContext &ir,
                    const CodeGenOptions &options = {},
                    const LoopHintMap &loopHints = {})
    -> llvm::Expected<mlir::OwningOpRef<mlir::ModuleOp>>;

} // namespace cmlir::frontend

#endif // CMLIR_FRONTEND_H
