#ifndef CLANG_MLIR_SOURCE_H
#define CLANG_MLIR_SOURCE_H

#include "mlir/IR/Location.h"
#include "clang/AST/ASTContext.h"
#include "llvm/ADT/StringRef.h"

namespace clang_mlir {

// A borrowed view: no AST pointers are retained in the generated module.
class Source {
public:
  Source(clang::ASTContext &ast, mlir::MLIRContext &ir) : ast(ast), ir(ir) {}

  auto location(clang::SourceLocation loc) const -> mlir::Location;
  auto error(clang::SourceLocation loc, llvm::StringRef message) -> void;

  class TraceScope {
  public:
    TraceScope(Source &source, llvm::StringRef name, clang::SourceLocation loc,
               bool enabled);
    ~TraceScope();
    TraceScope(const TraceScope &) = delete;
    auto operator=(const TraceScope &) -> TraceScope & = delete;

  private:
    Source &source;
    bool enabled;
  };

  auto failed() const -> bool {
    return hasError || ast.getDiagnostics().hasErrorOccurred();
  }

private:
  clang::ASTContext &ast;
  mlir::MLIRContext &ir;
  bool hasError = false;
  unsigned traceDepth = 0;
};

} // namespace clang_mlir

#endif // CLANG_MLIR_SOURCE_H
