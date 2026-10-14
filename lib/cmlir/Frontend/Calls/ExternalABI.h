#ifndef CMLIR_FRONTEND_EXTERNAL_ABI_H
#define CMLIR_FRONTEND_EXTERNAL_ABI_H

#include "CallLowering.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"

namespace cmlir::frontend {

auto getOrCreateLLVMFunctionDecl(mlir::OpBuilder &, mlir::ModuleOp,
                                 llvm::StringRef, mlir::LLVM::LLVMFunctionType)
    -> mlir::LLVM::LLVMFuncOp;
auto emitVariadicCall(mlir::OpBuilder &, mlir::Location, mlir::ModuleOp,
                      llvm::StringRef, llvm::ArrayRef<CallArg>,
                      mlir::FunctionType) -> mlir::FailureOr<mlir::Value>;

} // namespace cmlir::frontend

#endif // CMLIR_FRONTEND_EXTERNAL_ABI_H
