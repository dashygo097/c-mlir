#include "./ContextManager.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"

namespace cmlirc {

ContextManager::ContextManager(clang::ASTContext *clangCtx) {
  clang_context_ = clangCtx;
  mlir_context_ = std::make_unique<mlir::MLIRContext>();
  builder_ = std::make_unique<mlir::OpBuilder>(mlir_context_.get());

  // Load necessary dialects
  mlir_context_->getOrLoadDialect<mlir::func::FuncDialect>();
  mlir_context_->getOrLoadDialect<mlir::memref::MemRefDialect>();
  mlir_context_->getOrLoadDialect<mlir::arith::ArithDialect>();
  mlir_context_->getOrLoadDialect<mlir::scf::SCFDialect>();

  module_ = std::make_unique<mlir::ModuleOp>(
      mlir::ModuleOp::create(builder_->getUnknownLoc()));
}

void ContextManager::dump() {
  mlir::OpPrintingFlags flags;

  module_->print(llvm::outs(), flags);
}

} // namespace cmlirc
