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

bool CMLIRConverter::TraverseWhileStmt(clang::WhileStmt *whileStmt) {
  if (!currentFunc)
    return true;

  mlir::OpBuilder &builder = contextManager.Builder();
  mlir::Location loc = builder.getUnknownLoc();

  const bool hasBreak = detail::stmtHasBreakInLoop(whileStmt);
  const bool hasContinue = detail::stmtHasContinueInLoop(whileStmt);
  const bool hasReturn = detail::stmtHasReturnInLoop(whileStmt);
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
                                  detail::boolConst(builder, loc, false),
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
    mlir::Value initKeepGoing = detail::boolConst(builder, loc, true);

    auto whileOp = mlir::scf::WhileOp::create(
        builder, loc, mlir::TypeRange{funcRetType},
        mlir::ValueRange{initRetVal, initKeepGoing},
        [&](mlir::OpBuilder &b, mlir::Location l, mlir::ValueRange args) {
          mlir::OpBuilder::InsertionGuard outerGuard(builder);
          builder.setInsertionPointToEnd(b.getInsertionBlock());

          mlir::Value whileCond =
              detail::toBool(builder, loc, generateExpr(whileStmt->getCond()));

          mlir::Value finalCond =
              detail::andi(builder, loc, whileCond, args[1]);

          if (breakFlag) {
            mlir::Value notBroke = detail::noti(
                builder, loc,
                mlir::memref::LoadOp::create(builder, loc, breakFlag)
                    .getResult());
            finalCond = detail::andi(builder, loc, finalCond, notBroke);
          }

          mlir::scf::ConditionOp::create(builder, loc, finalCond,
                                         mlir::ValueRange{args[0]});
        },
        [&](mlir::OpBuilder &b, mlir::Location l, mlir::ValueRange) {
          mlir::scf::YieldOp::create(
              b, l, mlir::ValueRange{initRetVal, initKeepGoing});
        });

    mlir::Block *afterBlock = &whileOp.getAfter().front();
    afterBlock->back().erase(); // remove placeholder yield
    builder.setInsertionPointToEnd(afterBlock);

    assert(afterBlock->getNumArguments() == 1 &&
           "expected exactly the retval arg");
    mlir::Value prevRetVal = afterBlock->getArgument(0);

    mlir::Value iterRetFlag =
        mlir::memref::AllocaOp::create(
            builder, loc, mlir::MemRefType::get({}, builder.getI1Type()))
            .getResult();
    mlir::memref::StoreOp::create(builder, loc,
                                  detail::boolConst(builder, loc, false),
                                  iterRetFlag, mlir::ValueRange{});

    mlir::Value iterRetSlot =
        mlir::memref::AllocaOp::create(builder, loc,
                                       mlir::MemRefType::get({}, funcRetType))
            .getResult();

    loopStack.push_back(LoopContext{&whileOp.getBefore().front(), afterBlock,
                                    breakFlag, continueFlag, iterRetFlag,
                                    iterRetSlot});

    auto *compound =
        llvm::dyn_cast_or_null<clang::CompoundStmt>(whileStmt->getBody());
    if (needsGuard && compound) {
      for (clang::Stmt *s : compound->body()) {
        mlir::Value guard = detail::buildGuard(builder, loc, breakFlag,
                                               continueFlag, iterRetFlag);
        detail::emitGuarded(builder, loc, guard, [&] { TraverseStmt(s); });
      }
    } else {
      TraverseStmt(whileStmt->getBody());
    }

    loopStack.pop_back();

    if (continueFlag)
      mlir::memref::StoreOp::create(builder, loc,
                                    detail::boolConst(builder, loc, false),
                                    continueFlag, mlir::ValueRange{});

    mlir::Value didReturn =
        mlir::memref::LoadOp::create(builder, loc, iterRetFlag).getResult();

    mlir::Value loadedRetVal =
        mlir::memref::LoadOp::create(builder, loc, iterRetSlot).getResult();

    mlir::Value newRetVal =
        mlir::arith::SelectOp::create(builder, loc, didReturn, loadedRetVal,
                                      prevRetVal)
            .getResult();

    mlir::Value newKeepGoing = detail::noti(builder, loc, didReturn);

    mlir::scf::YieldOp::create(builder, loc,
                               mlir::ValueRange{newRetVal, newKeepGoing});

    builder.setInsertionPointAfter(whileOp);
    mlir::func::ReturnOp::create(builder, loc,
                                 mlir::ValueRange{whileOp.getResult(0)});
    return true;
  }

  auto whileOp = mlir::scf::WhileOp::create(
      builder, loc, mlir::TypeRange{}, mlir::ValueRange{},
      [&](mlir::OpBuilder &b, mlir::Location l, mlir::ValueRange) {
        mlir::OpBuilder::InsertionGuard g(builder);
        builder.setInsertionPointToEnd(b.getInsertionBlock());
        mlir::Value cond =
            detail::toBool(builder, loc, generateExpr(whileStmt->getCond()));
        if (breakFlag) {
          mlir::Value notBroke =
              detail::noti(builder, loc,
                           mlir::memref::LoadOp::create(builder, loc, breakFlag)
                               .getResult());
          cond = detail::andi(builder, loc, cond, notBroke);
        }
        mlir::scf::ConditionOp::create(builder, loc, cond, mlir::ValueRange{});
      },
      [&](mlir::OpBuilder &b, mlir::Location l, mlir::ValueRange) {
        mlir::scf::YieldOp::create(b, l, mlir::ValueRange{});
      });

  mlir::Block *afterBlock = &whileOp.getAfter().front();
  afterBlock->back().erase();
  builder.setInsertionPointToEnd(afterBlock);

  loopStack.push_back(LoopContext{&whileOp.getBefore().front(),
                                  afterBlock,
                                  breakFlag,
                                  continueFlag,
                                  {},
                                  {}});

  auto *compound =
      llvm::dyn_cast_or_null<clang::CompoundStmt>(whileStmt->getBody());
  if (needsGuard && compound) {
    for (clang::Stmt *s : compound->body()) {
      mlir::Value guard =
          detail::buildGuard(builder, loc, breakFlag, continueFlag, {});
      detail::emitGuarded(builder, loc, guard, [&] { TraverseStmt(s); });
    }
  } else {
    TraverseStmt(whileStmt->getBody());
  }

  loopStack.pop_back();

  if (continueFlag)
    mlir::memref::StoreOp::create(builder, loc,
                                  detail::boolConst(builder, loc, false),
                                  continueFlag, mlir::ValueRange{});

  detail::ensureYield(builder, loc, afterBlock);
  builder.setInsertionPointAfter(whileOp);
  return true;
}

} // namespace cmlirc
