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
  const bool hasContinue = detail::stmtHasContinueInLoop(doStmt);
  const bool needsGuard = hasBreak || hasContinue;

  mlir::Value breakFlag, continueFlag;

  auto allocFlag = [&](mlir::Value &flag) {
    mlir::Value falseVal = detail::boolConst(builder, loc, false);
    flag = mlir::memref::AllocaOp::create(
               builder, loc, mlir::MemRefType::get({}, builder.getI1Type()))
               .getResult();
    mlir::memref::StoreOp::create(builder, loc, falseVal, flag,
                                  mlir::ValueRange{});
  };

  if (hasBreak)
    allocFlag(breakFlag);
  if (hasContinue)
    allocFlag(continueFlag);

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
  beforeBlock->back().erase(); // remove the placeholder ConditionOp
  builder.setInsertionPointToEnd(beforeBlock);

  loopStack.push_back(
      {beforeBlock, &whileOp.getAfter().front(), breakFlag, continueFlag});

  auto *body = llvm::dyn_cast_or_null<clang::CompoundStmt>(doStmt->getBody());
  if (needsGuard && body) {
    for (clang::Stmt *s : body->body()) {
      mlir::Value shouldRun;
      if (hasBreak && hasContinue) {
        mlir::Value notBroke = detail::noti(
            builder, loc,
            mlir::memref::LoadOp::create(builder, loc, breakFlag).getResult());
        mlir::Value notCont = detail::noti(
            builder, loc,
            mlir::memref::LoadOp::create(builder, loc, continueFlag)
                .getResult());
        shouldRun = detail::andi(builder, loc, notBroke, notCont);
      } else if (hasBreak) {
        shouldRun = detail::noti(
            builder, loc,
            mlir::memref::LoadOp::create(builder, loc, breakFlag).getResult());
      } else {
        shouldRun = detail::noti(
            builder, loc,
            mlir::memref::LoadOp::create(builder, loc, continueFlag)
                .getResult());
      }

      auto ifOp = mlir::scf::IfOp::create(builder, loc, mlir::TypeRange{},
                                          shouldRun, /*hasElse=*/false);
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

  if (hasContinue) {
    mlir::memref::StoreOp::create(builder, loc,
                                  detail::boolConst(builder, loc, false),
                                  continueFlag, mlir::ValueRange{});
  }

  mlir::Value cond =
      doStmt->getCond()
          ? detail::toBool(builder, loc, generateExpr(doStmt->getCond()))
          : detail::boolConst(builder, loc, true);
  if (hasBreak) {
    mlir::Value notBroke = detail::noti(
        builder, loc,
        mlir::memref::LoadOp::create(builder, loc, breakFlag).getResult());
    cond = detail::andi(builder, loc, cond, notBroke);
  }
  mlir::scf::ConditionOp::create(builder, loc, cond, mlir::ValueRange{});

  if (hasContinue) {
    mlir::Block *afterBlock = &whileOp.getAfter().front();
    afterBlock->back().erase(); // remove placeholder YieldOp
    mlir::OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPointToEnd(afterBlock);
    mlir::memref::StoreOp::create(builder, loc,
                                  detail::boolConst(builder, loc, false),
                                  continueFlag, mlir::ValueRange{});
    mlir::scf::YieldOp::create(builder, loc, mlir::ValueRange{});
  }

  builder.setInsertionPointAfter(whileOp);
  return true;
}

} // namespace cmlirc
