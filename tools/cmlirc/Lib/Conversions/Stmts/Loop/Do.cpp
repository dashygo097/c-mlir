#include "../../../Converter.h"
#include "../../Utils/Casts.h"
#include "../../Utils/Constants.h"
#include "../../Utils/Numerics.h"
#include "../../Utils/StmtUtils.h"
#include "mlir/Dialect/SCF/IR/SCF.h"

namespace cmlirc {

bool CMLIRConverter::TraverseDoStmt(clang::DoStmt *doStmt) {
  if (!currentFunc)
    return true;

  mlir::OpBuilder &builder = context_manager_.Builder();
  mlir::Location loc = builder.getUnknownLoc();

  const bool hasBreak = detail::stmtHasBreakInLoop(doStmt);

  mlir::Value breakFlag;
  if (hasBreak) {
    mlir::Type i1 = builder.getI1Type();
    mlir::Value falseVal = detail::boolConst(builder, loc, false);
    breakFlag = mlir::memref::AllocaOp::create(builder, loc,
                                               mlir::MemRefType::get({}, i1))
                    .getResult();
    mlir::memref::StoreOp::create(builder, loc, falseVal, breakFlag,
                                  mlir::ValueRange{});
  }

  auto whileOp = mlir::scf::WhileOp::create(
      builder, loc, mlir::TypeRange{}, mlir::ValueRange{},
      [&](mlir::OpBuilder &b, mlir::Location l, mlir::ValueRange) {
        mlir::scf::ConditionOp::create(b, l, detail::boolConst(b, l, true),
                                       mlir::ValueRange{});
      },
      [&](mlir::OpBuilder &b, mlir::Location l, mlir::ValueRange) {
        mlir::scf::YieldOp::create(b, l, mlir::ValueRange{});
      });

  mlir::Block *beforeBlock = &whileOp.getBefore().front();
  beforeBlock->back().erase();
  builder.setInsertionPointToEnd(beforeBlock);

  loopStack.push_back({beforeBlock, &whileOp.getAfter().front(), breakFlag});

  auto *body = llvm::dyn_cast_or_null<clang::CompoundStmt>(doStmt->getBody());
  if (hasBreak && body) {
    for (clang::Stmt *s : body->body()) {
      mlir::Value broke =
          mlir::memref::LoadOp::create(builder, loc, breakFlag).getResult();
      mlir::Value notBroke = detail::noti(builder, loc, broke);
      auto ifOp = mlir::scf::IfOp::create(builder, loc, mlir::TypeRange{},
                                          notBroke, /*hasElse=*/false);
      {
        mlir::OpBuilder::InsertionGuard guard(builder);
        mlir::Block *thenBlk = &ifOp.getThenRegion().front();
        thenBlk->back().erase();
        builder.setInsertionPointToStart(thenBlk);
        TraverseStmt(s);
        builder.setInsertionPointToEnd(builder.getInsertionBlock());
        if (builder.getInsertionBlock()->empty() ||
            !builder.getInsertionBlock()
                 ->back()
                 .hasTrait<mlir::OpTrait::IsTerminator>())
          mlir::scf::YieldOp::create(builder, loc, mlir::ValueRange{});
      }
      builder.setInsertionPointAfter(ifOp);
    }
  } else {
    TraverseStmt(doStmt->getBody());
  }

  loopStack.pop_back();

  mlir::Value cond =
      doStmt->getCond()
          ? detail::toBool(builder, loc, generateExpr(doStmt->getCond()))
          : detail::boolConst(builder, loc, true);
  if (hasBreak) {
    mlir::Value broke =
        mlir::memref::LoadOp::create(builder, loc, breakFlag).getResult();
    mlir::Value notBroke = detail::noti(builder, loc, broke);
    cond = detail::andi(builder, loc, cond, notBroke);
  }
  mlir::scf::ConditionOp::create(builder, loc, cond, mlir::ValueRange{});

  builder.setInsertionPointAfter(whileOp);
  return true;
}

} // namespace cmlirc
