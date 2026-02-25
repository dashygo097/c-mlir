#include "../../../Converter.h"
#include "./CFUtils.h"
#include "mlir/Dialect/SCF/IR/SCF.h"

namespace cmlirc {

bool CMLIRConverter::TraverseWhileStmt(clang::WhileStmt *whileStmt) {
  if (!currentFunc)
    return true;

  mlir::OpBuilder &builder = context_manager_.Builder();
  mlir::Location loc = builder.getUnknownLoc();

  auto whileOp = mlir::scf::WhileOp::create(builder, loc, mlir::TypeRange{},
                                            mlir::ValueRange{});

  mlir::Block *beforeBlock = &whileOp.getBefore().front();
  {
    mlir::OpBuilder::InsertionGuard g(builder);
    builder.setInsertionPointToStart(beforeBlock);

    mlir::Value cond = convertToBool(generateExpr(whileStmt->getCond()));
    mlir::scf::ConditionOp::create(builder, loc, cond, mlir::ValueRange{});
  }

  mlir::Block *afterBlock = &whileOp.getAfter().front();
  builder.setInsertionPointToStart(afterBlock);

  loopStack_.push_back({beforeBlock, afterBlock});
  TraverseStmt(whileStmt->getBody());
  loopStack_.pop_back();

  detail::ensureYield(builder, loc, afterBlock);
  builder.setInsertionPointAfter(whileOp);

  return true;
}

} // namespace cmlirc
