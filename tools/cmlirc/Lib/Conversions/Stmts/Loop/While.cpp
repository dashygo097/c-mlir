#include "../../../Converter.h"
#include "../../Utils/Casts.h"
#include "../../Utils/Constants.h"
#include "../../Utils/Numerics.h"
#include "../../Utils/StmtUtils.h"
#include "./LoopUtils.h"
#include "mlir/Dialect/SCF/IR/SCF.h"

namespace cmlirc {

static mlir::Value buildGuard(mlir::OpBuilder &builder, mlir::Location loc,
                              mlir::Value breakFlag, mlir::Value continueFlag) {
  mlir::Value notBroke, notCont;

  if (breakFlag)
    notBroke = detail::noti(
        builder, loc,
        mlir::memref::LoadOp::create(builder, loc, breakFlag).getResult());

  if (continueFlag)
    notCont = detail::noti(
        builder, loc,
        mlir::memref::LoadOp::create(builder, loc, continueFlag).getResult());

  if (notBroke && notCont)
    return detail::andi(builder, loc, notBroke, notCont);
  return notBroke ? notBroke : notCont;
}

static void emitGuarded(mlir::OpBuilder &builder, mlir::Location loc,
                        mlir::Value guard, std::function<void()> emitBody) {
  auto ifOp = mlir::scf::IfOp::create(builder, loc, mlir::TypeRange{}, guard,
                                      /*hasElse=*/false);
  {
    mlir::OpBuilder::InsertionGuard g(builder);
    mlir::Block *thenBlk = &ifOp.getThenRegion().front();
    thenBlk->back().erase();
    builder.setInsertionPointToStart(thenBlk);
    emitBody();
    builder.setInsertionPointToEnd(builder.getInsertionBlock());
    if (builder.getInsertionBlock()->empty() ||
        !builder.getInsertionBlock()
             ->back()
             .hasTrait<mlir::OpTrait::IsTerminator>())
      mlir::scf::YieldOp::create(builder, loc, mlir::ValueRange{});
  }
  builder.setInsertionPointAfter(ifOp);
}

bool CMLIRConverter::TraverseWhileStmt(clang::WhileStmt *whileStmt) {
  if (!currentFunc)
    return true;

  mlir::OpBuilder &builder = context_manager_.Builder();
  mlir::Location loc = builder.getUnknownLoc();

  const bool hasBreak = detail::stmtHasBreakInLoop(whileStmt);
  const bool hasContinue = detail::stmtHasContinueInLoop(whileStmt);
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
        mlir::Value cond =
            detail::toBool(builder, loc, generateExpr(whileStmt->getCond()));
        if (hasBreak) {
          mlir::Value notBroke = detail::noti(
              b, l, mlir::memref::LoadOp::create(b, l, breakFlag).getResult());
          cond = detail::andi(builder, loc, cond, notBroke);
        }
        mlir::scf::ConditionOp::create(b, l, cond, mlir::ValueRange{});
      },
      [&](mlir::OpBuilder &b, mlir::Location l, mlir::ValueRange) {
        mlir::scf::YieldOp::create(b, l, mlir::ValueRange{});
      });

  mlir::Block *afterBlock = &whileOp.getAfter().front();
  afterBlock->back().erase();
  builder.setInsertionPointToEnd(afterBlock);

  loopStack.push_back(
      {&whileOp.getBefore().front(), afterBlock, breakFlag, continueFlag});

  auto *body =
      llvm::dyn_cast_or_null<clang::CompoundStmt>(whileStmt->getBody());
  if (needsGuard && body) {
    for (clang::Stmt *s : body->body()) {
      mlir::Value guard = buildGuard(builder, loc, breakFlag, continueFlag);
      emitGuarded(builder, loc, guard, [&] { TraverseStmt(s); });
    }
  } else {
    TraverseStmt(whileStmt->getBody());
  }

  loopStack.pop_back();

  if (hasContinue) {
    mlir::memref::StoreOp::create(builder, loc,
                                  detail::boolConst(builder, loc, false),
                                  continueFlag, mlir::ValueRange{});
  }

  detail::ensureYield(builder, loc, afterBlock);
  builder.setInsertionPointAfter(whileOp);
  return true;
}

} // namespace cmlirc
