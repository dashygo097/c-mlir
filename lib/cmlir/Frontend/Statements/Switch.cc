#include "../Lowering.h"
#include "clang-mlir/Casts.h"
#include "clang-mlir/Numerics.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "llvm/ADT/ScopeExit.h"

namespace cmlir::frontend {

auto ASTLowering::emitSwitch(clang::SwitchStmt *stmt) -> bool {
  if (stmt->getInit() && !emitStatement(stmt->getInit())) {
    return false;
  }
  if (stmt->getConditionVariable() &&
      !emitVariable(stmt->getConditionVariable())) {
    return false;
  }
  auto &builder = context.builder();
  auto loc = context.source().location(stmt->getSwitchLoc());
  auto emitted = expressions.emitValue(*stmt->getCond());
  if (mlir::failed(emitted)) {
    return false;
  }
  auto value = *emitted;
  if (!value || !mlir::isa<mlir::IntegerType>(value.getType())) {
    return false;
  }
  auto *entry = builder.getInsertionBlock();
  auto *region = entry->getParent();
  if (region != &function.operation.getBody()) {
    context.source().error(stmt->getSwitchLoc(),
                           "switch requires function control flow");
    return false;
  }
  auto *body = llvm::dyn_cast<clang::CompoundStmt>(stmt->getBody());
  if (!body) {
    context.source().error(stmt->getSwitchLoc(),
                           "switch body must be compound");
    return false;
  }

  struct Arm {
    clang::SwitchCase *label;
    mlir::Block *block;
    llvm::SmallVector<clang::Stmt *> statements;
  };

  llvm::SmallVector<Arm> arms;
  for (auto *child : body->body()) {
    while (auto *label = llvm::dyn_cast<clang::SwitchCase>(child)) {
      arms.push_back({label, builder.createBlock(region), {}});
      child = label->getSubStmt();
    }
    if (!arms.empty()) {
      arms.back().statements.push_back(child);
    }
  }
  auto *exit = builder.createBlock(region);
  auto *defaultBlock = exit;
  for (auto &arm : arms) {
    if (llvm::isa<clang::DefaultStmt>(arm.label)) {
      defaultBlock = arm.block;
    }
  }
  builder.setInsertionPointToEnd(entry);
  for (auto &arm : arms) {
    auto *label = llvm::dyn_cast<clang::CaseStmt>(arm.label);
    if (!label) {
      continue;
    }
    if (label->getRHS()) {
      context.source().error(label->getCaseLoc(),
                             "case ranges are unsupported");
      return false;
    }
    clang::Expr::EvalResult evaluated;
    if (!label->getLHS()->EvaluateAsInt(evaluated, context.ast())) {
      return false;
    }
    auto constant = mlir::arith::ConstantOp::create(
        builder, loc, value.getType(),
        builder.getIntegerAttr(value.getType(), evaluated.Val.getInt()));
    auto match = mlir::arith::CmpIOp::create(
        builder, loc, mlir::arith::CmpIPredicate::eq, value, constant);
    auto *next = builder.createBlock(region);
    builder.setInsertionPointToEnd(match->getBlock());
    mlir::cf::CondBranchOp::create(builder, loc, match, arm.block,
                                   mlir::ValueRange{}, next,
                                   mlir::ValueRange{});
    builder.setInsertionPointToStart(next);
  }
  mlir::cf::BranchOp::create(builder, loc, defaultBlock);
  function.controls.push_back({nullptr, exit});
  auto restore =
      llvm::scope_exit([&]() -> void { function.controls.pop_back(); });
  for (size_t i = 0; i < arms.size(); ++i) {
    builder.setInsertionPointToStart(arms[i].block);
    for (auto *child : arms[i].statements) {
      if (!emitStatement(child)) {
        return false;
      }
    }
    auto *tail = builder.getInsertionBlock();
    if (tail->empty() ||
        !tail->back().hasTrait<mlir::OpTrait::IsTerminator>()) {
      mlir::cf::BranchOp::create(
          builder, loc, i + 1 < arms.size() ? arms[i + 1].block : exit);
    }
  }
  builder.setInsertionPointToStart(exit);
  return true;
}

} // namespace cmlir::frontend
