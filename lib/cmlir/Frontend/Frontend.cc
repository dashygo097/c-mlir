#include "cmlir/Frontend/Frontend.h"
#include "Lowering.h"
#include "clang-support/Discovery.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Verifier.h"
#include "clang/Basic/TargetInfo.h"
#include "llvm/ADT/DenseSet.h"

namespace cmlir::frontend {

namespace {

auto collectReachable(clang::FunctionDecl *decl,
                      llvm::DenseSet<const clang::FunctionDecl *> &seen,
                      llvm::SmallVectorImpl<clang::FunctionDecl *> &ordered)
    -> void {
  if (!seen.insert(decl->getCanonicalDecl()).second) {
    return;
  }
  for (auto *callee : clang_support::directDefinedCallees(*decl)) {
    collectReachable(callee, seen, ordered);
  }
  ordered.push_back(decl);
}

} // namespace

auto registerDialects(mlir::DialectRegistry &registry) -> void {
  registry.insert<mlir::arith::ArithDialect, mlir::func::FuncDialect,
                  mlir::cf::ControlFlowDialect, mlir::memref::MemRefDialect,
                  mlir::LLVM::LLVMDialect, mlir::scf::SCFDialect,
                  mlir::affine::AffineDialect, mlir::math::MathDialect,
                  mlir::vector::VectorDialect>();
}

auto generateModule(clang::ASTContext &ast, mlir::MLIRContext &ir,
                    const CodeGenOptions &options, const LoopHintMap &loopHints)
    -> llvm::Expected<mlir::OwningOpRef<mlir::ModuleOp>> {
  if (ast.getTargetInfo().getPointerWidth(clang::LangAS::Default) != 64) {
    return llvm::createStringError(
        llvm::inconvertibleErrorCode(),
        "clang-to-mlir currently requires a 64-bit pointer ABI");
  }
  mlir::DialectRegistry registry;
  registerDialects(registry);
  ir.appendDialectRegistry(registry);
  ir.loadDialect<mlir::arith::ArithDialect, mlir::func::FuncDialect,
                 mlir::cf::ControlFlowDialect, mlir::memref::MemRefDialect,
                 mlir::LLVM::LLVMDialect, mlir::scf::SCFDialect,
                 mlir::affine::AffineDialect, mlir::math::MathDialect,
                 mlir::vector::VectorDialect>();
  mlir::OwningOpRef<mlir::ModuleOp> module =
      mlir::ModuleOp::create(mlir::UnknownLoc::get(&ir));
  clang_mlir::Context context(ast, module.get());
  ASTLowering lowering(context, loopHints);
  lowering.setTrace(options.traceLowering);
  bool irError = false;
  mlir::ScopedDiagnosticHandler diagnostics(
      &ir, [&](mlir::Diagnostic &diagnostic) -> mlir::LogicalResult {
        if (diagnostic.getSeverity() == mlir::DiagnosticSeverity::Error) {
          irError = true;
        }
        return mlir::failure(); // Keep normal diagnostic emission.
      });
  bool success = true;
  if (options.function.empty()) {
    success = lowering.emitDeclaration(ast.getTranslationUnitDecl());
  } else {
    auto definitions = clang_support::definedFunctions(ast);
    clang::FunctionDecl *target = nullptr;
    for (auto *decl : definitions) {
      if (decl->getNameAsString() != options.function &&
          decl->getQualifiedNameAsString() != options.function) {
        continue;
      }
      if (target && target->getCanonicalDecl() != decl->getCanonicalDecl()) {
        context.source().error(decl->getLocation(),
                               "ambiguous function selection");
        success = false;
        break;
      }
      target = decl;
    }
    if (success && !target) {
      context.source().error({}, "selected function definition not found");
      success = false;
    }
    if (success) {
      llvm::DenseSet<const clang::FunctionDecl *> seen;
      llvm::SmallVector<clang::FunctionDecl *> ordered;
      collectReachable(target, seen, ordered);
      for (auto *decl : ordered) {
        if (!lowering.emitFunction(decl)) {
          success = false;
          break;
        }
      }
    }
  }
  if (!success || lowering.failed() || irError ||
      mlir::failed(mlir::verify(module.get()))) {
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "AST lowering failed");
  }
  return std::move(module);
}

} // namespace cmlir::frontend
