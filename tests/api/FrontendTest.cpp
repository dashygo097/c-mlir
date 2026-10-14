#include "cmlir/Frontend/Frontend.h"
#include "clang-mlir/Address.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Verifier.h"
#include "clang/Tooling/Tooling.h"
#include "llvm/Support/raw_ostream.h"

namespace {

auto check(bool condition, llvm::StringRef message) -> bool {
  if (!condition) {
    llvm::errs() << "FAIL: " << message << '\n';
  }
  return condition;
}

} // namespace

auto main() -> int {
  mlir::DialectRegistry registry;
  cmlir::frontend::registerDialects(registry);
  mlir::MLIRContext ir(registry);
  auto ast = clang::tooling::buildASTFromCodeWithArgs(
      "int helper(int n) { if (n) ++n; return n; } "
      "int entry(int n) { return helper(n); }",
      {"-xc"}, "api.c");
  if (!check(bool(ast), "in-memory AST creation")) {
    return 1;
  }
  cmlir::frontend::CodeGenOptions options;
  options.function = "entry";
  auto first =
      cmlir::frontend::generateModule(ast->getASTContext(), ir, options);
  if (!first) {
    llvm::errs() << llvm::toString(first.takeError()) << '\n';
    return 1;
  }
  if (!check(bool(first->get().lookupSymbol<mlir::func::FuncOp>("helper")),
             "selected entry includes its defined callees")) {
    return 1;
  }
  if (!check(mlir::succeeded(mlir::verify(first->get())),
             "generated module verifies without an optimization library")) {
    return 1;
  }
  auto second =
      cmlir::frontend::generateModule(ast->getASTContext(), ir, options);
  if (!second) {
    llvm::consumeError(second.takeError());
    return 1;
  }
  if (!check(first->get() != second->get(),
             "compilations own independent modules")) {
    return 1;
  }

  // A library caller gets a recoverable error, never process termination.
  options.function = "missing";
  auto missing =
      cmlir::frontend::generateModule(ast->getASTContext(), ir, options);
  if (!check(!missing, "missing entry returns Error")) {
    return 1;
  }
  llvm::consumeError(missing.takeError());

  mlir::OpBuilder builder(&ir);
  // Address validation rejects invalid ranks without emitting malformed ops.
  builder.setInsertionPointToStart(
      &first->get()
           .lookupSymbol<mlir::func::FuncOp>("helper")
           .getBody()
           .front());
  auto constant = mlir::arith::ConstantIntOp::create(
      builder, builder.getUnknownLoc(), 1, 32);
  auto slot =
      clang_mlir::createScalarSlot(builder, builder.getUnknownLoc(), constant);
  auto invalid =
      clang_mlir::Address::get(slot, builder.getI32Type(), {constant});
  if (!check(mlir::failed(invalid), "invalid address fails validation")) {
    return 1;
  }
  llvm::outs() << "CMLIRFrontend API checks passed\n";
  return 0;
}
