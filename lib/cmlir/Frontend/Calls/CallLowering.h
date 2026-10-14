#ifndef CMLIR_FRONTEND_CALL_LOWERING_H
#define CMLIR_FRONTEND_CALL_LOWERING_H

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "clang/AST/Decl.h"

namespace cmlir::frontend {

struct CallArg {
  mlir::Value value;
  clang::QualType clangType;
};

auto functionSymbolName(const clang::FunctionDecl &) -> std::string;
auto getOrCreateFunctionDecl(mlir::OpBuilder &, mlir::ModuleOp,
                             const std::string &, mlir::FunctionType)
    -> mlir::func::FuncOp;
auto getOrCreateLLVMFunctionDecl(mlir::OpBuilder &, mlir::ModuleOp,
                                 llvm::StringRef, mlir::LLVM::LLVMFunctionType)
    -> mlir::LLVM::LLVMFuncOp;
auto coerceFuncCallArg(mlir::OpBuilder &, mlir::Location, const CallArg &,
                       mlir::Type) -> mlir::Value;
auto emitExternalCall(mlir::OpBuilder &, mlir::Location, mlir::ModuleOp,
                      llvm::StringRef, llvm::ArrayRef<CallArg>,
                      llvm::ArrayRef<mlir::Type>, mlir::Type, bool,
                      mlir::Value &) -> bool;
auto tryEmitStdlibCall(mlir::OpBuilder &, mlir::Location, mlir::ModuleOp,
                       llvm::StringRef, llvm::ArrayRef<CallArg>, mlir::Value &)
    -> bool;

} // namespace cmlir::frontend

#endif // CMLIR_FRONTEND_CALL_LOWERING_H
