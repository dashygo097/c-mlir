// WhileStmt.cpp
#include "../../../Converter.h"
#include "mlir/Dialect/SCF/IR/SCF.h"

namespace cmlirc {

bool CMLIRConverter::TraverseWhileStmt(clang::WhileStmt *whileStmt) {
  if (!currentFunc)
    return true;

  mlir::OpBuilder &b = context_manager_.Builder();
  mlir::Location loc = b.getUnknownLoc();

  auto whileOp =
      mlir::scf::WhileOp::create(b, loc, mlir::TypeRange{}, mlir::ValueRange{});

  mlir::Block *beforeBlock = &whileOp.getBefore().front();
  {
    mlir::OpBuilder::InsertionGuard g(b);
    b.setInsertionPointToStart(beforeBlock);

    mlir::Value cond = convertToBool(generateExpr(whileStmt->getCond()));
    mlir::scf::ConditionOp::create(b, loc, cond, mlir::ValueRange{});
  }

  mlir::Block *afterBlock = &whileOp.getAfter().front();
  b.setInsertionPointToStart(afterBlock);

  loopStack_.push_back({beforeBlock, afterBlock});
  TraverseStmt(whileStmt->getBody());
  loopStack_.pop_back();

  ensureYield(b, loc, afterBlock);
  b.setInsertionPointAfter(whileOp);

  return true;
}

} // namespace cmlirc
