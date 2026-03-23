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

ContextManager::ContextManager(clang::ASTContext *clangContext,
                               mlir::DialectRegistry *registry) {
  clangCtx = clangContext;
  mlirCtx = std::make_unique<mlir::MLIRContext>();
  builder = std::make_unique<mlir::OpBuilder>(mlirCtx.get());

  if (registry)
    mlirCtx->appendDialectRegistry(*registry);

  // Load necessary dialects
  mlirCtx->getOrLoadDialect<mlir::LLVM::LLVMDialect>();
  mlirCtx->getOrLoadDialect<mlir::func::FuncDialect>();
  mlirCtx->getOrLoadDialect<mlir::memref::MemRefDialect>();
  mlirCtx->getOrLoadDialect<mlir::arith::ArithDialect>();
  mlirCtx->getOrLoadDialect<mlir::cf::ControlFlowDialect>();
  mlirCtx->getOrLoadDialect<mlir::scf::SCFDialect>();
  mlirCtx->getOrLoadDialect<mlir::affine::AffineDialect>();
  mlirCtx->getOrLoadDialect<mlir::math::MathDialect>();

  module = std::make_unique<mlir::ModuleOp>(
      mlir::ModuleOp::create(builder->getUnknownLoc()));
}

void ContextManager::dump(llvm::raw_ostream &os) {
  std::string buf;
  llvm::raw_string_ostream ss(buf);
  mlir::OpPrintingFlags flags;
  module->print(ss, flags);
  ss.flush();
  llvm::StringRef ir(buf);

  while (!ir.empty()) {
    // Comments → gray (#8b949e)
    if (ir.starts_with("//")) {
      auto end = ir.find('\n');
      size_t len = (end == llvm::StringRef::npos ? ir.size() : end);
      llvm::WithColor(os, llvm::raw_ostream::WHITE, /*bold=*/false)
          << ir.slice(0, len);
      ir = ir.drop_front(len);
      continue;
    }

    // Strings → light blue (#a5d6ff)
    if (ir[0] == '"') {
      auto end = ir.find('"', 1);
      size_t len = (end == llvm::StringRef::npos ? ir.size() : end + 1);
      llvm::WithColor(os, llvm::raw_ostream::BLUE, /*bold=*/false)
          << ir.slice(0, len);
      ir = ir.drop_front(len);
      continue;
    }

    // Numbers → blue (#79c0ff)
    if (std::isdigit((unsigned char)ir[0])) {
      size_t len = 0;
      while (len < ir.size() && (std::isdigit((unsigned char)ir[len]) ||
                                 ir[len] == '.' || ir[len] == '_'))
        ++len;
      llvm::WithColor(os, llvm::raw_ostream::CYAN, /*bold=*/false)
          << ir.slice(0, len);
      ir = ir.drop_front(len);
      continue;
    }

    // Identifiers / keywords / types
    if (std::isalpha((unsigned char)ir[0]) || ir[0] == '_') {
      size_t len = 0;
      while (len < ir.size() && (std::isalnum((unsigned char)ir[len]) ||
                                 ir[len] == '_' || ir[len] == '.'))
        ++len;
      llvm::StringRef word = ir.slice(0, len);

      if (word.contains('.')) {
        // dialect ops → purple (#d2a8ff)
        llvm::WithColor(os, llvm::raw_ostream::MAGENTA, /*bold=*/true) << word;
      } else if (word == "func" || word == "return" || word == "module" ||
                 word == "do" || word == "default" || word == "case" ||
                 word == "to" || word == "step" || word == "iter_args") {
        // keywords → red (#ff7b72)
        llvm::WithColor(os, llvm::raw_ostream::RED, /*bold=*/true) << word;
      } else if (word == "i1" || word == "i8" || word == "i16" ||
                 word == "i32" || word == "i64" || word == "f32" ||
                 word == "f64" || word == "index" || word == "memref" ||
                 word == "ptr" || word == "none" || word == "vararg") {
        // types → blue (#79c0ff)
        llvm::WithColor(os, llvm::raw_ostream::CYAN, /*bold=*/false) << word;
      } else {
        os << word;
      }
      ir = ir.drop_front(len);
      continue;
    }

    // SSA values %foo → default text (white bold)
    if (ir[0] == '%') {
      size_t len = 1;
      while (len < ir.size() && (std::isalnum((unsigned char)ir[len]) ||
                                 ir[len] == '_' || ir[len] == '#'))
        ++len;
      llvm::WithColor(os, llvm::raw_ostream::WHITE, /*bold=*/true)
          << ir.slice(0, len);
      ir = ir.drop_front(len);
      continue;
    }

    // Symbol refs @foo → orange (#ffa657)
    if (ir[0] == '@') {
      size_t len = 1;
      while (len < ir.size() &&
             (std::isalnum((unsigned char)ir[len]) || ir[len] == '_'))
        ++len;
      llvm::WithColor(os, llvm::raw_ostream::YELLOW, /*bold=*/true)
          << ir.slice(0, len);
      ir = ir.drop_front(len);
      continue;
    }

    // Block labels ^bb0 → red
    if (ir[0] == '^') {
      size_t len = 1;
      while (len < ir.size() &&
             (std::isalnum((unsigned char)ir[len]) || ir[len] == '_'))
        ++len;
      llvm::WithColor(os, llvm::raw_ostream::RED, /*bold=*/false)
          << ir.slice(0, len);
      ir = ir.drop_front(len);
      continue;
    }

    os << ir[0];
    ir = ir.drop_front(1);
  }
}

void ContextManager::dump() { dump(llvm::outs()); }

} // namespace cmlirc
