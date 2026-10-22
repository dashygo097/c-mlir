#ifndef CMLIR_FRONTEND_MODULE_STATE_H
#define CMLIR_FRONTEND_MODULE_STATE_H

#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "clang/AST/Decl.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"

namespace cmlir::frontend {

struct ModuleState {
  llvm::DenseSet<const clang::FunctionDecl *> emittedFunctions;
};

} // namespace cmlir::frontend

#endif // CMLIR_FRONTEND_MODULE_STATE_H
