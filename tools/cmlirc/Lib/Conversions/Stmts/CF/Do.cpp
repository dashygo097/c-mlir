#include "../../../Converter.h"
#include "./CFUtils.h"
#include "mlir/Dialect/SCF/IR/SCF.h"

namespace cmlirc {

bool CMLIRConverter::TraverseDoStmt(clang::DoStmt *doStmt) {
  if (!currentFunc)
    return true;

  mlir::OpBuilder &builder = context_manager_.Builder();
  mlir::Location loc = builder.getUnknownLoc();

  auto whileOp = mlir::scf::WhileOp::create(builder, loc, mlir::TypeRange{},
                                            mlir::ValueRange{});

  mlir::Block *beforeBlock = &whileOp.getBefore().front();
  builder.setInsertionPointToStart(beforeBlock);

  loopStack_.push_back({beforeBlock, &whileOp.getAfter().front()});
  TraverseStmt(doStmt->getBody());
  loopStack_.pop_back();

  {
    builder.setInsertionPointToEnd(beforeBlock);
    mlir::Value cond = convertToBool(generateExpr(doStmt->getCond()));
    mlir::scf::ConditionOp::create(builder, loc, cond, mlir::ValueRange{});
  }

  mlir::Block *afterBlock = &whileOp.getAfter().front();
  detail::ensureYield(builder, loc, afterBlock);

  builder.setInsertionPointAfter(whileOp);
  return true;
}

} // namespace cmlirc
