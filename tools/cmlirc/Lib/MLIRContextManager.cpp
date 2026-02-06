#include "./MLIRContextManager.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"

namespace cmlirc {

MLIRContextManager::MLIRContextManager() {
  context_ = std::make_unique<mlir::MLIRContext>();
  builder_ = std::make_unique<mlir::OpBuilder>(context_.get());

  // Load necessary dialects
  context_->getOrLoadDialect<mlir::func::FuncDialect>();
  context_->getOrLoadDialect<mlir::memref::MemRefDialect>();
  context_->getOrLoadDialect<mlir::arith::ArithDialect>();
  context_->getOrLoadDialect<mlir::scf::SCFDialect>();

  module_ = std::make_unique<mlir::ModuleOp>(
      mlir::ModuleOp::create(builder_->getUnknownLoc()));
}

void MLIRContextManager::dump() {
  mlir::OpPrintingFlags flags;

  module_->print(llvm::outs(), flags);
}

} // namespace cmlirc
