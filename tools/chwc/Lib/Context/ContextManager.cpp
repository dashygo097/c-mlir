#include "./ContextManager.h"
#include "circt/Dialect/Comb/CombDialect.h"
#include "circt/Dialect/HW/HWDialect.h"
#include "circt/Dialect/SV/SVDialect.h"
#include "circt/Dialect/Seq/SeqDialect.h"
#include "mlir/IR/Builders.h"
#include "llvm/Support/WithColor.h"
#include "llvm/Support/raw_ostream.h"
#include <cctype>
#include <string>

namespace chwc {

CHWContextManager::CHWContextManager(clang::ASTContext *clangContext,
                                     mlir::DialectRegistry *registry)
    : clangCtx(clangContext) {

  mlirCtx = std::make_unique<mlir::MLIRContext>();
  builder = std::make_unique<mlir::OpBuilder>(mlirCtx.get());

  if (registry) {
    mlirCtx->appendDialectRegistry(*registry);
  }

  // Load necessary dialects
  mlirCtx->getOrLoadDialect<circt::hw::HWDialect>();
  mlirCtx->getOrLoadDialect<circt::comb::CombDialect>();
  mlirCtx->getOrLoadDialect<circt::seq::SeqDialect>();
  mlirCtx->getOrLoadDialect<circt::sv::SVDialect>();

  module = mlir::ModuleOp::create(builder->getUnknownLoc());
}

void CHWContextManager::dump(llvm::raw_ostream &os) {
  std::string buf;
  llvm::raw_string_ostream ss(buf);
  mlir::OpPrintingFlags flags;
  module->print(ss, flags);

  ss.flush();

  llvm::StringRef ir(buf);

  while (!ir.empty()) {
    // Comments → gray
    if (ir.starts_with("//")) {
      auto end = ir.find('\n');
      size_t len = (end == llvm::StringRef::npos ? ir.size() : end);
      llvm::WithColor(os, llvm::raw_ostream::WHITE, /*bold=*/false)
          << ir.slice(0, len);
      ir = ir.drop_front(len);
      continue;
    }

    // Strings → light blue
    if (ir[0] == '"') {
      auto end = ir.find('"', 1);
      size_t len = (end == llvm::StringRef::npos ? ir.size() : end + 1);
      llvm::WithColor(os, llvm::raw_ostream::BLUE, /*bold=*/false)
          << ir.slice(0, len);
      ir = ir.drop_front(len);
      continue;
    }

    // Numbers → cyan
    if (std::isdigit(static_cast<unsigned char>(ir[0]))) {
      size_t len = 0;
      while (len < ir.size() &&
             (std::isdigit(static_cast<unsigned char>(ir[len])) ||
              ir[len] == '.' || ir[len] == '_')) {
        ++len;
      }
      llvm::WithColor(os, llvm::raw_ostream::CYAN, /*bold=*/false)
          << ir.slice(0, len);
      ir = ir.drop_front(len);
      continue;
    }

    // Identifiers / keywords / types
    if (std::isalpha(static_cast<unsigned char>(ir[0])) || ir[0] == '_') {
      size_t len = 0;
      while (len < ir.size() &&
             (std::isalnum(static_cast<unsigned char>(ir[len])) ||
              ir[len] == '_' || ir[len] == '.')) {
        ++len;
      }
      llvm::StringRef word = ir.slice(0, len);

      if (word.contains('.')) {
        // dialect ops → purple
        llvm::WithColor(os, llvm::raw_ostream::MAGENTA, /*bold=*/true) << word;
      } else if (word == "func" || word == "return" || word == "module" ||
                 word == "do" || word == "default" || word == "case" ||
                 word == "to" || word == "step" || word == "iter_args") {
        // keywords → red
        llvm::WithColor(os, llvm::raw_ostream::RED, /*bold=*/true) << word;
      } else if (word == "i1" || word == "i8" || word == "i16" ||
                 word == "i32" || word == "i64" || word == "f32" ||
                 word == "f64" || word == "index" || word == "memref" ||
                 word == "ptr" || word == "none" || word == "vararg") {
        // types → cyan
        llvm::WithColor(os, llvm::raw_ostream::CYAN, /*bold=*/false) << word;
      } else {
        os << word;
      }
      ir = ir.drop_front(len);
      continue;
    }

    // SSA values %foo → white bold
    if (ir[0] == '%') {
      size_t len = 1;
      while (len < ir.size() &&
             (std::isalnum(static_cast<unsigned char>(ir[len])) ||
              ir[len] == '_' || ir[len] == '#')) {
        ++len;
      }
      llvm::WithColor(os, llvm::raw_ostream::WHITE, /*bold=*/true)
          << ir.slice(0, len);
      ir = ir.drop_front(len);
      continue;
    }

    // Symbol refs @foo → yellow
    if (ir[0] == '@') {
      size_t len = 1;
      while (len < ir.size() &&
             (std::isalnum(static_cast<unsigned char>(ir[len])) ||
              ir[len] == '_')) {
        ++len;
      }
      llvm::WithColor(os, llvm::raw_ostream::YELLOW, /*bold=*/true)
          << ir.slice(0, len);
      ir = ir.drop_front(len);
      continue;
    }

    // Block labels ^bb0 → red
    if (ir[0] == '^') {
      size_t len = 1;
      while (len < ir.size() &&
             (std::isalnum(static_cast<unsigned char>(ir[len])) ||
              ir[len] == '_')) {
        ++len;
      }
      llvm::WithColor(os, llvm::raw_ostream::RED, /*bold=*/false)
          << ir.slice(0, len);
      ir = ir.drop_front(len);
      continue;
    }

    os << ir[0];
    ir = ir.drop_front(1);
  }
}

void CHWContextManager::dump() { dump(llvm::outs()); }

} // namespace chwc
