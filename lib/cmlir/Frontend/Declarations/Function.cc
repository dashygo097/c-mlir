#include "../Calls/CallLowering.h"
#include "../Lowering.h"
#include "clang-mlir/Address.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "clang/Basic/SourceManager.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/ScopeExit.h"

namespace cmlir::frontend {

namespace {

auto references(const clang::Stmt *stmt, const clang::VarDecl *var) -> bool {
  if (!stmt) {
    return false;
  }
  if (auto *ref = llvm::dyn_cast<clang::DeclRefExpr>(stmt)) {
    if (ref->getDecl() == var) {
      return true;
    }
  }
  for (auto *child : stmt->children()) {
    if (references(child, var)) {
      return true;
    }
  }
  return false;
}

auto needsStorage(const clang::Stmt *stmt, const clang::VarDecl *var) -> bool {
  if (!stmt) {
    return false;
  }
  if (auto *op = llvm::dyn_cast<clang::BinaryOperator>(stmt)) {
    if (op->isAssignmentOp() && references(op->getLHS(), var)) {
      return true;
    }
  }
  if (auto *op = llvm::dyn_cast<clang::UnaryOperator>(stmt)) {
    if ((op->isIncrementDecrementOp() || op->getOpcode() == clang::UO_AddrOf) &&
        references(op->getSubExpr(), var)) {
      return true;
    }
  }
  if (llvm::isa<clang::CallExpr>(stmt) && references(stmt, var)) {
    return true;
  }
  for (auto *child : stmt->children()) {
    if (needsStorage(child, var)) {
      return true;
    }
  }
  return false;
}

auto reachableBlocks(mlir::Block *entry) -> llvm::DenseSet<mlir::Block *> {
  llvm::DenseSet<mlir::Block *> reachable;
  llvm::SmallVector<mlir::Block *> pending{entry};
  while (!pending.empty()) {
    auto *block = pending.pop_back_val();
    if (!reachable.insert(block).second || block->empty()) {
      continue;
    }
    if (block->back().hasTrait<mlir::OpTrait::IsTerminator>()) {
      for (auto *successor : block->back().getSuccessors()) {
        pending.push_back(successor);
      }
    }
  }
  return reachable;
}

} // namespace

auto ASTLowering::emitFunction(clang::FunctionDecl *decl) -> bool {
  if (decl->isImplicit() || !decl->getDefinition()) {
    return true;
  }

  decl = decl->getDefinition();
  if (llvm::isa<clang::CXXConstructorDecl, clang::CXXDestructorDecl>(decl)) {
    context.source().error(
        decl->getLocation(),
        "constructor and destructor definitions are unsupported");
    return false;
  }
  if (auto *method = llvm::dyn_cast<clang::CXXMethodDecl>(decl)) {
    if (method->isVirtual()) {
      context.source().error(decl->getLocation(),
                             "virtual methods are unsupported");
      return false;
    }
  }
  if (!module.emittedFunctions.insert(decl->getCanonicalDecl()).second) {
    return true;
  }

  clang_mlir::Source::TraceScope traceFrame(
      context.source(), decl->getNameAsString(), decl->getLocation(), trace);

  // skip system headers
  clang::SourceManager &sm = context.ast().getSourceManager();
  if (sm.isInSystemHeader(decl->getLocation())) {
    return true;
  }
  if (decl->isVariadic()) {
    context.source().error(decl->getLocation(),
                           "variadic function definitions are unsupported");
    return false;
  }

  mlir::OpBuilder &builder = context.builder();
  mlir::OpBuilder::InsertionGuard insertion(builder);
  builder.setInsertionPointToEnd(context.module().getBody());

  // Name of the function
  std::string funcName = functionSymbolName(*decl);

  bool isInstanceMethod = false;
  if (auto *method = llvm::dyn_cast<clang::CXXMethodDecl>(decl)) {
    isInstanceMethod = !method->isStatic();
  }
  auto signature =
      lowerFunctionSignature(types, *decl, CallConvention::Internal);
  if (mlir::failed(signature)) {
    context.source().error(decl->getLocation(),
                           "unsupported function definition signature");
    return false;
  }
  auto returnTypes = signature->getResults();
  auto funcOp =
      getOrCreateFunctionDecl(builder, context.module(), funcName, *signature);
  if (!funcOp) {
    return false;
  }
  if (!funcOp.isDeclaration()) {
    return true;
  }
  funcOp->moveBefore(context.module().getBody(),
                     context.module().getBody()->end());
  funcOp.setPublic();

  // Create entry block
  auto *entryBlock = funcOp.addEntryBlock();
  builder.setInsertionPointToStart(entryBlock);

  auto savedFunction = std::move(function);
  function = FunctionState{};
  function.operation = funcOp;
  auto reset =
      llvm::scope_exit([&]() -> void { function = std::move(savedFunction); });

  uint32_t argIdx = 0;
  if (isInstanceMethod) {
    function.thisValue = entryBlock->getArgument(argIdx++);
  } else {
    function.thisValue = nullptr;
  }

  // Map parameters to block arguments
  for (uint32_t i = 0; i < decl->getNumParams(); ++i) {
    auto *param = decl->getParamDecl(i);
    auto value = entryBlock->getArgument(argIdx++);
    if ((value.getType().isIntOrFloat() || value.getType().isIndex()) &&
        needsStorage(decl->getBody(), param)) {
      function.locals[param] =
          clang_mlir::createScalarSlot(builder, builder.getUnknownLoc(), value);
    } else if (mlir::isa<mlir::LLVM::LLVMStructType>(value.getType()) &&
               needsStorage(decl->getBody(), param)) {
      auto one = mlir::arith::ConstantIntOp::create(
          builder, builder.getUnknownLoc(), 1, 64);
      auto slot = mlir::LLVM::AllocaOp::create(
          builder, builder.getUnknownLoc(),
          mlir::LLVM::LLVMPointerType::get(builder.getContext()),
          value.getType(), one);
      mlir::LLVM::StoreOp::create(builder, builder.getUnknownLoc(), value,
                                  slot);
      function.locals[param] = slot;
    } else {
      function.parameters[param] = value;
    }
  }

  // Traverse function body manually
  if (!emitStatement(decl->getBody())) {
    return false;
  }

  mlir::Block *finalBlock = builder.getInsertionBlock();
  auto reachable = reachableBlocks(entryBlock);
  bool finalReachable = reachable.contains(finalBlock);
  llvm::SmallVector<mlir::Block *> dead;
  for (auto &block : funcOp.getBody()) {
    if (!reachable.contains(&block)) {
      dead.push_back(&block);
    }
  }
  for (auto *block : dead) {
    block->dropAllReferences();
  }
  for (auto *block : dead) {
    block->erase();
  }
  if (!finalReachable) {
    builder.clearInsertionPoint();
    return true;
  }
  builder.setInsertionPointToEnd(finalBlock);

  // Ensure the function has a terminator
  if (finalBlock->empty() ||
      !finalBlock->back().hasTrait<mlir::OpTrait::IsTerminator>()) {
    if (!returnTypes.empty()) {
      // A merge block with no predecessors is unreachable (e.g. both if arms
      // return).
      if (finalBlock != entryBlock && finalBlock->hasNoPredecessors()) {
        finalBlock->erase();
        return true;
      }
      if (decl->isMain()) {
        auto zero = mlir::arith::ConstantIntOp::create(
            builder, builder.getUnknownLoc(), 0, 32);
        mlir::func::ReturnOp::create(builder, builder.getUnknownLoc(),
                                     zero.getResult());
      } else {
        context.source().error(
            decl->getLocation(),
            "non-void function can fall through without a return");
        return false;
      }
    } else {
      mlir::func::ReturnOp::create(builder, builder.getUnknownLoc());
    }
  }

  return true;
}

} // namespace cmlir::frontend
