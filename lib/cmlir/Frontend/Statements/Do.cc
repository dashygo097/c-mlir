#include "../ControlFlow/LoopAnalysis.h"
#include "../Lowering.h"
#include "clang-mlir/Casts.h"
#include "mlir/Dialect/SCF/IR/SCF.h"

namespace cmlir::frontend {

auto ASTLowering::emitDo(clang::DoStmt *stmt) -> bool {
  if (control::hasControlExit(stmt->getBody())) {
    return emitCFGLoop(stmt->getCond(), stmt->getBody(), nullptr, true);
  }
  auto &builder = context.builder();
  auto loc = context.source().location(stmt->getBeginLoc());
  auto loop = mlir::scf::WhileOp::create(builder, loc, mlir::TypeRange{},
                                         mlir::ValueRange{});
  builder.createBlock(&loop.getBefore());
  if (!emitStatement(stmt->getBody())) {
    return false;
  }
  auto cond = clang_mlir::toBool(builder, loc, generateExpr(stmt->getCond()));
  if (!cond) {
    return false;
  }
  mlir::scf::ConditionOp::create(builder, loc, cond, mlir::ValueRange{});
  builder.createBlock(&loop.getAfter());
  mlir::scf::YieldOp::create(builder, loc);
  builder.setInsertionPointAfter(loop);
  return true;
}

} // namespace cmlir::frontend
