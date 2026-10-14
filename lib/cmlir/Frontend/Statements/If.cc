#include "../Lowering.h"
#include "clang-mlir/Casts.h"
#include "clang-mlir/Regions.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"

namespace cmlir::frontend {

auto ASTLowering::emitIf(clang::IfStmt *stmt) -> bool {
  if (stmt->getInit() && !emitStatement(stmt->getInit())) {
    return false;
  }
  if (stmt->getConditionVariable() &&
      !emitVariable(stmt->getConditionVariable())) {
    return false;
  }
  auto &builder = context.builder();
  auto loc = context.source().location(stmt->getIfLoc());
  auto condition =
      clang_mlir::toBool(builder, loc, generateExpr(stmt->getCond()));
  if (!condition) {
    return false;
  }
  auto *entry = builder.getInsertionBlock();
  auto *region = entry->getParent();
  auto arm = [&](mlir::Block &block, clang::Stmt *body,
                 mlir::Block *continuation) -> bool {
    return mlir::succeeded(clang_mlir::emitArm(
        builder, loc, block, continuation, [&]() -> mlir::LogicalResult {
          return mlir::success(emitStatement(body));
        }));
  };
  if (region != &function.operation.getBody()) {
    auto branch = mlir::scf::IfOp::create(
        builder, loc, mlir::TypeRange{}, condition, stmt->getElse() != nullptr);
    if (!arm(*branch.thenBlock(), stmt->getThen(), nullptr)) {
      return false;
    }
    if (stmt->getElse() &&
        !arm(*branch.elseBlock(), stmt->getElse(), nullptr)) {
      return false;
    }
    builder.setInsertionPointAfter(branch);
    return true;
  }
  auto *thenBlock = builder.createBlock(region);
  auto *elseBlock = stmt->getElse() ? builder.createBlock(region) : nullptr;
  auto *merge = builder.createBlock(region);
  builder.setInsertionPointToEnd(entry);
  mlir::cf::CondBranchOp::create(
      builder, loc, condition, thenBlock, mlir::ValueRange{},
      elseBlock ? elseBlock : merge, mlir::ValueRange{});
  if (!arm(*thenBlock, stmt->getThen(), merge)) {
    return false;
  }
  if (elseBlock && !arm(*elseBlock, stmt->getElse(), merge)) {
    return false;
  }
  builder.setInsertionPointToStart(merge);
  return true;
}

} // namespace cmlir::frontend
