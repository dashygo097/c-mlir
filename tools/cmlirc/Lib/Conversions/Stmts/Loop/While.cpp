#include "../../../Converter.h"
#include "../../Utils/Casts.h"
#include "./LoopUtils.h"
#include "mlir/Dialect/SCF/IR/SCF.h"

namespace cmlirc {

bool CMLIRConverter::TraverseWhileStmt(clang::WhileStmt *whileStmt) {
  if (!currentFunc)
    return true;

  mlir::OpBuilder &builder = context_manager_.Builder();
  mlir::Location loc = builder.getUnknownLoc();

  auto beforBuilder = [&](mlir::OpBuilder &b, mlir::Location l,
                          mlir::ValueRange args) {
    mlir::Value cond =
        detail::toBool(builder, loc, generateExpr(whileStmt->getCond()));
    mlir::scf::ConditionOp::create(b, l, cond, mlir::ValueRange{});
  };

  auto afterBuilder = [&](mlir::OpBuilder &b, mlir::Location l,
                          mlir::ValueRange args) {
    mlir::scf::YieldOp::create(b, l);
  };

  auto whileOp = mlir::scf::WhileOp::create(builder, loc, mlir::TypeRange{},
                                            mlir::ValueRange{}, beforBuilder,
                                            afterBuilder);

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
