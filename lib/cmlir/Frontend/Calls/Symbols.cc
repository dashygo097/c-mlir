#include "../Lowering.h"
#include "CallLowering.h"
#include "clang-mlir/Casts.h"
#include "clang-mlir/Constants.h"
#include "clang-mlir/MemrefABI.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "clang/AST/DeclCXX.h"
#include "clang/AST/Mangle.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/WithColor.h"

#include "mlir/IR/Diagnostics.h"

namespace cmlir::frontend {

auto getOrCreateFunctionDecl(mlir::OpBuilder &builder, mlir::ModuleOp module,
                             const std::string &name,
                             mlir::FunctionType funcType)
    -> mlir::func::FuncOp {
  if (auto existing = module.lookupSymbol<mlir::func::FuncOp>(name)) {
    if (existing.getFunctionType() != funcType) {
      mlir::emitError(builder.getUnknownLoc(),
                      "conflicting function symbol signature");
      return {};
    }
    return existing;
  }

  mlir::OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPointToStart(module.getBody());

  auto funcOp = mlir::func::FuncOp::create(builder, builder.getUnknownLoc(),
                                           name, funcType);
  funcOp.setPrivate();

  return funcOp;
}

auto getOrCreateLLVMFunctionDecl(mlir::OpBuilder &builder,
                                 mlir::ModuleOp module, llvm::StringRef name,
                                 mlir::LLVM::LLVMFunctionType funcType)
    -> mlir::LLVM::LLVMFuncOp {
  if (auto existing = module.lookupSymbol<mlir::LLVM::LLVMFuncOp>(name)) {
    if (existing.getFunctionType() != funcType) {
      mlir::emitError(builder.getUnknownLoc(),
                      "conflicting function symbol signature");
      return {};
    }
    return existing;
  }

  mlir::OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPointToStart(module.getBody());

  return mlir::LLVM::LLVMFuncOp::create(builder, builder.getUnknownLoc(), name,
                                        funcType,
                                        mlir::LLVM::Linkage::External);
}

auto functionSymbolName(const clang::FunctionDecl &decl) -> std::string {
  if (decl.isExternC() || decl.isMain()) {
    return decl.getNameAsString();
  }
  llvm::DenseSet<const clang::FunctionDecl *> overloads;
  for (auto *named : decl.getDeclContext()->lookup(decl.getDeclName())) {
    if (auto *function = llvm::dyn_cast<clang::FunctionDecl>(named)) {
      overloads.insert(function->getCanonicalDecl());
    }
  }
  if (!decl.hasBody() || overloads.size() > 1 ||
      !decl.getDeclContext()->isTranslationUnit()) {
    auto mangler = std::unique_ptr<clang::MangleContext>(
        decl.getASTContext().createMangleContext());
    if (mangler->shouldMangleDeclName(&decl)) {
      std::string name;
      llvm::raw_string_ostream stream(name);
      mangler->mangleName(clang::GlobalDecl(&decl), stream);
      return name;
    }
  }
  return decl.getNameAsString();
}

} // namespace cmlir::frontend
