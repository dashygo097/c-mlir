#include "../../../Converter.h"
#include "../../Utils/Constants.h"
#include "mlir/Dialect/SCF/IR/SCF.h"

namespace cmlirc {

bool CMLIRConverter::TraverseDoStmt(clang::DoStmt *doStmt) {
  if (!currentFunc)
    return true;

  mlir::OpBuilder &builder = context_manager_.Builder();
  mlir::Location loc = builder.getUnknownLoc();

  auto whileOp = mlir::scf::WhileOp::create(
      builder, loc, mlir::TypeRange{}, mlir::ValueRange{},
      [&](mlir::OpBuilder &b, mlir::Location l, mlir::ValueRange args) {
        mlir::scf::ConditionOp::create(b, l, detail::boolConst(b, l, true),
                                       mlir::ValueRange{});
      },
      [&](mlir::OpBuilder &b, mlir::Location l, mlir::ValueRange args) {
        mlir::scf::YieldOp::create(b, l, mlir::ValueRange{});
      });

  mlir::Block *beforeBlock = &whileOp.getBefore().front();
  beforeBlock->back().erase();

  builder.setInsertionPointToEnd(beforeBlock);
  loopStack.push_back({beforeBlock, &whileOp.getAfter().front()});
  TraverseStmt(doStmt->getBody());
  loopStack.pop_back();

  mlir::Value cond = doStmt->getCond()
                         ? convertToBool(generateExpr(doStmt->getCond()))
                         : detail::boolConst(builder, loc, true);
  mlir::scf::ConditionOp::create(builder, loc, cond, mlir::ValueRange{});

  builder.setInsertionPointAfter(whileOp);
  return true;
}

} // namespace cmlirc
