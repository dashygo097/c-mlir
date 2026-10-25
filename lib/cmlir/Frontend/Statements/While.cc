#include "../ControlFlow/LoopAnalysis.h"
#include "../Lowering.h"
#include "clang-mlir/Casts.h"
#include "mlir/Dialect/SCF/IR/SCF.h"

namespace cmlir::frontend {

auto ASTLowering::emitWhile(clang::WhileStmt *stmt) -> bool {
  if (control::hasControlExit(stmt->getBody())) {
    return emitCFGLoop(stmt->getCond(), stmt->getBody(), nullptr, false);
  }
  auto &builder = context.builder();
  auto loc = context.source().location(stmt->getBeginLoc());
  auto loop = mlir::scf::WhileOp::create(builder, loc, mlir::TypeRange{},
                                         mlir::ValueRange{});
  builder.createBlock(&loop.getBefore());
  auto value = expressions.emitValue(*stmt->getCond());
  if (mlir::failed(value)) {
    return false;
  }
  auto cond = clang_mlir::toBool(builder, loc, *value);
  if (!cond) {
    return false;
  }
  mlir::scf::ConditionOp::create(builder, loc, cond, mlir::ValueRange{});
  builder.createBlock(&loop.getAfter());
  if (!emitStatement(stmt->getBody())) {
    return false;
  }
  mlir::scf::YieldOp::create(builder, loc);
  builder.setInsertionPointAfter(loop);
  return true;
}

} // namespace cmlir::frontend
