#include "clang-mlir/Source.h"
#include "mlir/IR/MLIRContext.h"
#include "clang/Basic/Diagnostic.h"
#include "clang/Basic/SourceManager.h"

namespace clang_mlir {

auto Source::location(clang::SourceLocation loc) const -> mlir::Location {
  if (loc.isInvalid()) {
    return mlir::UnknownLoc::get(&ir);
  }
  auto &sm = ast.getSourceManager();
  auto presumed = sm.getPresumedLoc(sm.getExpansionLoc(loc));
  if (presumed.isInvalid()) {
    return mlir::UnknownLoc::get(&ir);
  }
  return mlir::FileLineColLoc::get(&ir, presumed.getFilename(),
                                   presumed.getLine(), presumed.getColumn());
}

auto Source::error(clang::SourceLocation loc, llvm::StringRef message) -> void {
  hasError = true;
  auto &diags = ast.getDiagnostics();
  unsigned id = diags.getCustomDiagID(clang::DiagnosticsEngine::Error, "%0");
  diags.Report(loc, id) << message;
}

Source::TraceScope::TraceScope(Source &source, llvm::StringRef name,
                               clang::SourceLocation loc, bool enabled)
    : source(source), enabled(enabled) {
  if (!enabled) {
    return;
  }
  llvm::errs().indent(source.traceDepth * 2)
      << name << " at " << source.location(loc) << '\n';
  ++source.traceDepth;
}

Source::TraceScope::~TraceScope() {
  if (enabled) {
    --source.traceDepth;
  }
}

} // namespace clang_mlir
