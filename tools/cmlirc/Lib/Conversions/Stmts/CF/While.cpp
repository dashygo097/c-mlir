#include "../../../Converter.h"
#include "./CFUtils.h"
#include "mlir/Dialect/SCF/IR/SCF.h"

namespace cmlirc {

bool CMLIRConverter::TraverseWhileStmt(clang::WhileStmt *whileStmt) {
  if (!currentFunc)
    return true;

  mlir::OpBuilder &builder = context_manager_.Builder();
  mlir::Location loc = builder.getUnknownLoc();

  auto whileOp = mlir::scf::WhileOp::create(
      builder, loc, mlir::TypeRange{}, mlir::ValueRange{},
      [&](mlir::OpBuilder &builder, mlir::Location loc, mlir::ValueRange args) {
        mlir::Value cond = convertToBool(generateExpr(whileStmt->getCond()));
        mlir::scf::ConditionOp::create(builder, loc, cond, mlir::ValueRange{});
      },
      [&](mlir::OpBuilder &builder, mlir::Location loc, mlir::ValueRange args) {
        mlir::scf::YieldOp::create(builder, loc, mlir::ValueRange{});
      });

  mlir::Block *afterBlock = &whileOp.getAfter().front();

  afterBlock->back().erase();

  builder.setInsertionPointToEnd(afterBlock);
  loopStack.push_back({&whileOp.getBefore().front(), afterBlock});
  TraverseStmt(whileStmt->getBody());
  loopStack.pop_back();

  detail::ensureYield(builder, loc, afterBlock);
  builder.setInsertionPointAfter(whileOp);

  return true;
}

} // namespace cmlirc
