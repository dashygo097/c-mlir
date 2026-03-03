#include "./ContextManager.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlow.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"
#include "llvm/Support/WithColor.h"

namespace cmlirc {

ContextManager::ContextManager(clang::ASTContext *clangCtx) {
  clang_context_ = clangCtx;
  mlir_context_ = std::make_unique<mlir::MLIRContext>();
  builder_ = std::make_unique<mlir::OpBuilder>(mlir_context_.get());

  // Load necessary dialects
  mlir_context_->getOrLoadDialect<mlir::LLVM::LLVMDialect>();
  mlir_context_->getOrLoadDialect<mlir::func::FuncDialect>();
  mlir_context_->getOrLoadDialect<mlir::memref::MemRefDialect>();
  mlir_context_->getOrLoadDialect<mlir::arith::ArithDialect>();
  mlir_context_->getOrLoadDialect<mlir::cf::ControlFlowDialect>();
  mlir_context_->getOrLoadDialect<mlir::scf::SCFDialect>();
  mlir_context_->getOrLoadDialect<mlir::affine::AffineDialect>();
  mlir_context_->getOrLoadDialect<mlir::math::MathDialect>();

  module_ = std::make_unique<mlir::ModuleOp>(
      mlir::ModuleOp::create(builder_->getUnknownLoc()));
}

void ContextManager::dump(llvm::raw_ostream &out) {
  std::string buf;
  llvm::raw_string_ostream ss(buf);
  mlir::OpPrintingFlags flags;
  module_->print(ss, flags);
  ss.flush();

  llvm::StringRef ir(buf);
  while (!ir.empty()) {
    if (ir.starts_with("//")) {
      auto end = ir.find('\n');
      llvm::WithColor(out, llvm::raw_ostream::GREEN)
          << ir.slice(0, end == llvm::StringRef::npos ? ir.size() : end);
      ir = ir.drop_front(end == llvm::StringRef::npos ? ir.size() : end);
      continue;
    }
    if (ir[0] == '"') {
      auto end = ir.find('"', 1);
      size_t len = (end == llvm::StringRef::npos ? ir.size() : end + 1);
      llvm::WithColor(out, llvm::raw_ostream::GREEN) << ir.slice(0, len);
      ir = ir.drop_front(len);
      continue;
    }
    if (std::isdigit((unsigned char)ir[0])) {
      size_t len = 0;
      while (len < ir.size() && (std::isdigit((unsigned char)ir[len]) ||
                                 ir[len] == '.' || ir[len] == '_'))
        ++len;
      llvm::WithColor(out, llvm::raw_ostream::MAGENTA) << ir.slice(0, len);
      ir = ir.drop_front(len);
      continue;
    }
    if (std::isalpha((unsigned char)ir[0]) || ir[0] == '_') {
      size_t len = 0;
      while (len < ir.size() && (std::isalnum((unsigned char)ir[len]) ||
                                 ir[len] == '_' || ir[len] == '.'))
        ++len;
      llvm::StringRef word = ir.slice(0, len);

      if (word.contains('.')) {
        llvm::WithColor(out, llvm::raw_ostream::YELLOW, /*bold=*/true) << word;
      } else if (word == "func" || word == "return" || word == "module" ||
                 word == "do" || word == "default" || word == "case" ||
                 word == "to" || word == "step" || word == "iter_args") {
        llvm::WithColor(out, llvm::raw_ostream::BLUE, /*bold=*/true) << word;
      } else if (word == "i1" || word == "i8" || word == "i16" ||
                 word == "i32" || word == "i64" || word == "f32" ||
                 word == "f64" || word == "index" || word == "memref" ||
                 word == "ptr" || word == "none" || word == "vararg") {
        llvm::WithColor(out, llvm::raw_ostream::CYAN) << word;
      } else {
        out << word;
      }
      ir = ir.drop_front(len);
      continue;
    }
    if (ir[0] == '%') {
      size_t len = 1;
      while (len < ir.size() && (std::isalnum((unsigned char)ir[len]) ||
                                 ir[len] == '_' || ir[len] == '#'))
        ++len;
      llvm::WithColor(out, llvm::raw_ostream::WHITE, /*bold=*/true)
          << ir.slice(0, len);
      ir = ir.drop_front(len);
      continue;
    }
    if (ir[0] == '@') {
      size_t len = 1;
      while (len < ir.size() &&
             (std::isalnum((unsigned char)ir[len]) || ir[len] == '_'))
        ++len;
      llvm::WithColor(out, llvm::raw_ostream::MAGENTA, /*bold=*/true)
          << ir.slice(0, len);
      ir = ir.drop_front(len);
      continue;
    }
    if (ir[0] == '^') {
      size_t len = 1;
      while (len < ir.size() &&
             (std::isalnum((unsigned char)ir[len]) || ir[len] == '_'))
        ++len;
      llvm::WithColor(out, llvm::raw_ostream::RED) << ir.slice(0, len);
      ir = ir.drop_front(len);
      continue;
    }
    out << ir[0];
    ir = ir.drop_front(1);
  }
}

void ContextManager::dump() { dump(llvm::outs()); }

} // namespace cmlirc
