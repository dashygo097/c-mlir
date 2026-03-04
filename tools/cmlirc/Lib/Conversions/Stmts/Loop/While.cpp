#include "../../../Converter.h"
#include "../../Utils/Casts.h"
#include "../../Utils/Numerics.h"
#include "./LoopUtils.h"
#include "mlir/Dialect/SCF/IR/SCF.h"

namespace cmlirc {

bool CMLIRConverter::TraverseWhileStmt(clang::WhileStmt *whileStmt) {
  if (!currentFunc)
    return true;

  mlir::OpBuilder &builder = context_manager_.Builder();
  mlir::Location loc = builder.getUnknownLoc();

  mlir::Type i1 = builder.getI1Type();
  mlir::Value falseValue = detail::boolConst(builder, loc, false);
  mlir::Value breakFlag = mlir::memref::AllocaOp::create(
                              builder, loc, mlir::MemRefType::get({}, i1))
                              .getResult();

  mlir::memref::StoreOp::create(builder, loc, falseValue, breakFlag,
                                mlir::ValueRange{});

  auto beforeBuilder = [&](mlir::OpBuilder &b, mlir::Location l,
                           mlir::ValueRange args) {
    mlir::Value cond =
        detail::toBool(builder, loc, generateExpr(whileStmt->getCond()));
    mlir::Value broke =
        mlir::memref::LoadOp::create(b, l, breakFlag).getResult();
    mlir::Value notBroke = detail::noti(builder, loc, broke);
    mlir::Value proceed =
        mlir::arith::AndIOp::create(b, l, cond, notBroke).getResult();
    mlir::scf::ConditionOp::create(b, l, proceed, mlir::ValueRange{});
  };

  auto afterBuilder = [&](mlir::OpBuilder &b, mlir::Location l,
                          mlir::ValueRange args) {
    mlir::scf::YieldOp::create(b, l, mlir::ValueRange{});
  };

  auto whileOp = mlir::scf::WhileOp::create(builder, loc, mlir::TypeRange{},
                                            mlir::ValueRange{}, beforeBuilder,
                                            afterBuilder);

  mlir::Block *afterBlock = &whileOp.getAfter().front();
  afterBlock->back().erase();
  builder.setInsertionPointToEnd(afterBlock);

  loopStack.push_back({&whileOp.getBefore().front(), afterBlock, breakFlag});

  auto *body =
      llvm::dyn_cast_or_null<clang::CompoundStmt>(whileStmt->getBody());
  if (body) {
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
    TraverseStmt(whileStmt->getBody());
  }

  loopStack.pop_back();

  detail::ensureYield(builder, loc, afterBlock);
  builder.setInsertionPointAfter(whileOp);

  return true;
}

} // namespace cmlirc
