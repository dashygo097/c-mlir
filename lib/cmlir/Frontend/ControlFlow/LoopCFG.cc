#include "../Lowering.h"
#include "clang-mlir/Casts.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "llvm/ADT/ScopeExit.h"

namespace cmlir::frontend {

auto ASTLowering::emitCFGLoop(clang::Expr *condition, clang::Stmt *body,
                              clang::Expr *increment, bool bodyFirst) -> bool {
  auto &builder = context.builder();
  auto loc = context.source().location(body->getBeginLoc());
  auto *entry = builder.getInsertionBlock();
  auto *region = entry->getParent();
  if (region != &function.operation.getBody()) {
    context.source().error(
        body->getBeginLoc(),
        "unstructured loop cannot escape a structured region");
    return false;
  }
  auto *header = builder.createBlock(region);
  auto *bodyBlock = builder.createBlock(region);
  auto *latch = builder.createBlock(region);
  auto *exit = builder.createBlock(region);
  builder.setInsertionPointToEnd(entry);
  mlir::cf::BranchOp::create(builder, loc, bodyFirst ? bodyBlock : header);
  builder.setInsertionPointToStart(header);
  mlir::Value cond;
  if (condition) {
    auto value = expressions.emitValue(*condition);
    if (mlir::failed(value)) {
      return false;
    }
    cond = clang_mlir::toBool(builder, loc, *value);
  } else {
    cond = clang_mlir::boolConst(builder, loc, true);
  }
  if (!cond) {
    return false;
  }
  mlir::cf::CondBranchOp::create(builder, loc, cond, bodyBlock,
                                 mlir::ValueRange{}, exit, mlir::ValueRange{});
  function.controls.push_back({latch, exit});
  auto restore =
      llvm::scope_exit([&]() -> void { function.controls.pop_back(); });
  builder.setInsertionPointToStart(bodyBlock);
  if (!emitStatement(body)) {
    return false;
  }
  auto *tail = builder.getInsertionBlock();
  if (tail->empty() || !tail->back().hasTrait<mlir::OpTrait::IsTerminator>()) {
    mlir::cf::BranchOp::create(builder, loc, latch);
  }
  builder.setInsertionPointToStart(latch);
  if (increment && mlir::failed(expressions.emitDiscarded(*increment))) {
    return false;
  }
  mlir::cf::BranchOp::create(builder, loc, header);
  builder.setInsertionPointToStart(exit);
  return true;
}

} // namespace cmlir::frontend
