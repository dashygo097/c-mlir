#include "../../../Converter.h"
#include "../../Utils/Casts.h"
#include "../../Utils/Constants.h"
#include "../../Utils/Numerics.h"
#include "../../Utils/StmtUtils.h"
#include "./LoopUtils.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"

namespace cmlirc {

bool CMLIRConverter::TraverseDoStmt(clang::DoStmt *doStmt) {
  if (!currentFunc)
    return true;

  mlir::OpBuilder &builder = contextManager.Builder();
  mlir::Location loc = builder.getUnknownLoc();

  const bool hasBreak = utils::stmtHasBreakInLoop(doStmt);
  const bool hasContinue = utils::stmtHasContinueInLoop(doStmt);
  const bool hasReturn = utils::stmtHasReturnInLoop(doStmt);
  const bool needsGuard = hasBreak || hasContinue || hasReturn;

  mlir::Type funcRetType;
  {
    auto res = currentFunc.getFunctionType().getResults();
    if (!res.empty())
      funcRetType = res[0];
  }

  auto allocBool = [&]() -> mlir::Value {
    auto a = mlir::memref::AllocaOp::create(
        builder, loc, mlir::MemRefType::get({}, builder.getI1Type()));
    mlir::memref::StoreOp::create(builder, loc,
                                  utils::boolConst(builder, loc, false),
                                  a.getResult(), mlir::ValueRange{});
    return a.getResult();
  };

  mlir::Value breakFlag = hasBreak ? allocBool() : mlir::Value{};
  mlir::Value continueFlag = hasContinue ? allocBool() : mlir::Value{};

  if (hasReturn && funcRetType) {
    mlir::Value initRetVal =
        mlir::arith::ConstantOp::create(builder, loc, funcRetType,
                                        builder.getZeroAttr(funcRetType))
            .getResult();
    mlir::Value initKeepGoing = utils::boolConst(builder, loc, true);

    auto whileOp = mlir::scf::WhileOp::create(
        builder, loc, mlir::TypeRange{funcRetType},
        mlir::ValueRange{initRetVal, initKeepGoing},
        [&](mlir::OpBuilder &b, mlir::Location l, mlir::ValueRange args) {
          mlir::scf::ConditionOp::create(b, l, args[1],
                                         mlir::ValueRange{args[0]});
        },
        [&](mlir::OpBuilder &b, mlir::Location l, mlir::ValueRange) {
          mlir::scf::YieldOp::create(
              b, l, mlir::ValueRange{initRetVal, initKeepGoing});
        });

    mlir::Block *afterBlock = &whileOp.getAfter().front();
    afterBlock->back().erase();
    builder.setInsertionPointToEnd(afterBlock);

    mlir::Value prevRetVal = afterBlock->getArgument(0);

    mlir::Value iterRetFlag =
        mlir::memref::AllocaOp::create(
            builder, loc, mlir::MemRefType::get({}, builder.getI1Type()))
            .getResult();
    mlir::memref::StoreOp::create(builder, loc,
                                  utils::boolConst(builder, loc, false),
                                  iterRetFlag, mlir::ValueRange{});

    mlir::Value iterRetSlot =
        mlir::memref::AllocaOp::create(builder, loc,
                                       mlir::MemRefType::get({}, funcRetType))
            .getResult();

    loopStack.push_back(LoopContext{&whileOp.getBefore().front(), afterBlock,
                                    breakFlag, continueFlag, iterRetFlag,
                                    iterRetSlot});

    auto *compound =
        llvm::dyn_cast_or_null<clang::CompoundStmt>(doStmt->getBody());
    if (needsGuard && compound) {
      for (clang::Stmt *s : compound->body()) {
        mlir::Value guard = utils::buildGuard(builder, loc, breakFlag,
                                              continueFlag, iterRetFlag);
        utils::emitGuarded(builder, loc, guard, [&] { TraverseStmt(s); });
      }
    } else {
      TraverseStmt(doStmt->getBody());
    }

    loopStack.pop_back();

    if (continueFlag)
      mlir::memref::StoreOp::create(builder, loc,
                                    utils::boolConst(builder, loc, false),
                                    continueFlag, mlir::ValueRange{});

    mlir::Value didReturn =
        mlir::memref::LoadOp::create(builder, loc, iterRetFlag).getResult();
    mlir::Value loadedRetVal =
        mlir::memref::LoadOp::create(builder, loc, iterRetSlot).getResult();
    mlir::Value newRetVal =
        mlir::arith::SelectOp::create(builder, loc, didReturn, loadedRetVal,
                                      prevRetVal)
            .getResult();

    mlir::Value origCond =
        doStmt->getCond()
            ? utils::toBool(builder, loc, generateExpr(doStmt->getCond()))
            : utils::boolConst(builder, loc, false);
    mlir::Value notDidReturn = utils::noti(builder, loc, didReturn);
    mlir::Value newKeepGoing =
        utils::andi(builder, loc, origCond, notDidReturn);
    if (breakFlag) {
      mlir::Value notBroke = utils::noti(
          builder, loc,
          mlir::memref::LoadOp::create(builder, loc, breakFlag).getResult());
      newKeepGoing = utils::andi(builder, loc, newKeepGoing, notBroke);
    }

    mlir::scf::YieldOp::create(builder, loc,
                               mlir::ValueRange{newRetVal, newKeepGoing});

    builder.setInsertionPointAfter(whileOp);
    mlir::func::ReturnOp::create(builder, loc,
                                 mlir::ValueRange{whileOp.getResult(0)});
    return true;
  } else if (hasReturn && !funcRetType) {
    auto whileOp = mlir::scf::WhileOp::create(
        builder, loc, mlir::TypeRange{}, mlir::ValueRange{},
        [&](mlir::OpBuilder &b, mlir::Location l, mlir::ValueRange args) {
          mlir::scf::ConditionOp::create(b, l, utils::boolConst(b, l, false),
                                         mlir::ValueRange{});
        },
        [&](mlir::OpBuilder &b, mlir::Location l, mlir::ValueRange) {
          mlir::scf::YieldOp::create(b, l, mlir::ValueRange{});
        });

    mlir::Block *beforeBlock = &whileOp.getBefore().front();
    beforeBlock->back().erase();
    builder.setInsertionPointToEnd(beforeBlock);

    mlir::Value iterRetFlag =
        mlir::memref::AllocaOp::create(
            builder, loc, mlir::MemRefType::get({}, builder.getI1Type()))
            .getResult();
    mlir::memref::StoreOp::create(builder, loc,
                                  utils::boolConst(builder, loc, false),
                                  iterRetFlag, mlir::ValueRange{});

    loopStack.push_back(LoopContext{beforeBlock,
                                    &whileOp.getAfter().front(),
                                    breakFlag,
                                    continueFlag,
                                    iterRetFlag,
                                    {}});

    auto *compound =
        llvm::dyn_cast_or_null<clang::CompoundStmt>(doStmt->getBody());
    if (needsGuard && compound) {
      for (clang::Stmt *s : compound->body()) {
        mlir::Value guard = utils::buildGuard(builder, loc, breakFlag,
                                              continueFlag, iterRetFlag);
        utils::emitGuarded(builder, loc, guard, [&] { TraverseStmt(s); });
      }
    } else {
      TraverseStmt(doStmt->getBody());
    }

    loopStack.pop_back();

    if (continueFlag)
      mlir::memref::StoreOp::create(builder, loc,
                                    utils::boolConst(builder, loc, false),
                                    continueFlag, mlir::ValueRange{});

    mlir::Value cond =
        doStmt->getCond()
            ? utils::toBool(builder, loc, generateExpr(doStmt->getCond()))
            : utils::boolConst(builder, loc, true);

    if (breakFlag) {
      mlir::Value notBroke = utils::noti(
          builder, loc,
          mlir::memref::LoadOp::create(builder, loc, breakFlag).getResult());
      cond = utils::andi(builder, loc, cond, notBroke);
    }
    if (iterRetFlag) {
      mlir::Value notRet = utils::noti(
          builder, loc,
          mlir::memref::LoadOp::create(builder, loc, iterRetFlag).getResult());
      cond = utils::andi(builder, loc, cond, notRet);
    }

    mlir::scf::ConditionOp::create(builder, loc, cond, mlir::ValueRange{});

    if (hasContinue) {
      mlir::Block *afterBlock = &whileOp.getAfter().front();
      afterBlock->back().erase();
      mlir::OpBuilder::InsertionGuard g(builder);
      builder.setInsertionPointToEnd(afterBlock);
      mlir::memref::StoreOp::create(builder, loc,
                                    utils::boolConst(builder, loc, false),
                                    continueFlag, mlir::ValueRange{});
      mlir::scf::YieldOp::create(builder, loc, mlir::ValueRange{});
    }

    builder.setInsertionPointAfter(whileOp);
    return true;
  }

  auto whileOp = mlir::scf::WhileOp::create(
      builder, loc, mlir::TypeRange{}, mlir::ValueRange{},
      [&](mlir::OpBuilder &b, mlir::Location l, mlir::ValueRange) {
        mlir::scf::ConditionOp::create(b, l, utils::boolConst(b, l, true),
                                       mlir::ValueRange{});
      },
      [&](mlir::OpBuilder &b, mlir::Location l, mlir::ValueRange) {
        mlir::scf::YieldOp::create(b, l, mlir::ValueRange{});
      });

  mlir::Block *beforeBlock = &whileOp.getBefore().front();
  beforeBlock->back().erase();
  builder.setInsertionPointToEnd(beforeBlock);

  loopStack.push_back(LoopContext{beforeBlock,
                                  &whileOp.getAfter().front(),
                                  breakFlag,
                                  continueFlag,
                                  {},
                                  {}});

  auto *compound =
      llvm::dyn_cast_or_null<clang::CompoundStmt>(doStmt->getBody());
  if (needsGuard && compound) {
    for (clang::Stmt *s : compound->body()) {
      mlir::Value guard =
          utils::buildGuard(builder, loc, breakFlag, continueFlag, {});
      utils::emitGuarded(builder, loc, guard, [&] { TraverseStmt(s); });
    }
  } else {
    TraverseStmt(doStmt->getBody());
  }

  loopStack.pop_back();

  if (continueFlag)
    mlir::memref::StoreOp::create(builder, loc,
                                  utils::boolConst(builder, loc, false),
                                  continueFlag, mlir::ValueRange{});

  mlir::Value cond =
      doStmt->getCond()
          ? utils::toBool(builder, loc, generateExpr(doStmt->getCond()))
          : utils::boolConst(builder, loc, true);
  if (breakFlag) {
    mlir::Value notBroke = utils::noti(
        builder, loc,
        mlir::memref::LoadOp::create(builder, loc, breakFlag).getResult());
    cond = utils::andi(builder, loc, cond, notBroke);
  }
  mlir::scf::ConditionOp::create(builder, loc, cond, mlir::ValueRange{});

  if (hasContinue) {
    mlir::Block *afterBlock = &whileOp.getAfter().front();
    afterBlock->back().erase();
    mlir::OpBuilder::InsertionGuard g(builder);
    builder.setInsertionPointToEnd(afterBlock);
    mlir::memref::StoreOp::create(builder, loc,
                                  utils::boolConst(builder, loc, false),
                                  continueFlag, mlir::ValueRange{});
    mlir::scf::YieldOp::create(builder, loc, mlir::ValueRange{});
  }

  builder.setInsertionPointAfter(whileOp);
  return true;
}

} // namespace cmlirc
