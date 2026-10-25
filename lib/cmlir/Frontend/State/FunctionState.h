#ifndef CMLIR_FRONTEND_FUNCTION_STATE_H
#define CMLIR_FRONTEND_FUNCTION_STATE_H

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "clang/AST/Decl.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"

namespace cmlir::frontend {

struct LoopContext {
  mlir::Block *continueBlock = nullptr;
  mlir::Block *exitBlock = nullptr;
};

struct FunctionState {
  llvm::DenseMap<const clang::VarDecl *, mlir::Value> locals;
  llvm::DenseMap<const clang::ParmVarDecl *, mlir::Value> parameters;
  mlir::func::FuncOp operation;
  const clang::FunctionDecl *declaration = nullptr;
  mlir::Value thisValue;
  llvm::SmallVector<LoopContext, 4> controls;
};

} // namespace cmlir::frontend

#endif // CMLIR_FRONTEND_FUNCTION_STATE_H
