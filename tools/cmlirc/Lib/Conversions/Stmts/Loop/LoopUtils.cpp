#include "./LoopUtils.h"
#include "../../Utils/Casts.h"
#include "../../Utils/Constants.h"
#include "../../Utils/Numerics.h"
#include "mlir/Dialect/SCF/IR/SCF.h"

namespace cmlirc::detail {

mlir::Value buildGuard(mlir::OpBuilder &builder, mlir::Location loc,
                       mlir::Value breakFlag, mlir::Value continueFlag,
                       mlir::Value returnFlag) {
  mlir::Value result;
  for (mlir::Value flag : {breakFlag, continueFlag, returnFlag}) {
    if (!flag)
      continue;
    mlir::Value notFlag = detail::noti(
        builder, loc,
        mlir::memref::LoadOp::create(builder, loc, flag).getResult());
    result = result ? detail::andi(builder, loc, result, notFlag) : notFlag;
  }
  return result;
}

void emitGuarded(mlir::OpBuilder &builder, mlir::Location loc,
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

void ensureYield(mlir::OpBuilder &builder, mlir::Location loc,
                 mlir::Block *block) {
  if (block->empty() ||
      !block->back().hasTrait<mlir::OpTrait::IsTerminator>()) {
    mlir::OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPointToEnd(block);
    mlir::scf::YieldOp::create(builder, loc, mlir::ValueRange{});
  }
}

bool classifyCondOp(clang::BinaryOperatorKind op, bool &isIncrementing) {
  using CBO = clang::BinaryOperatorKind;
  switch (op) {
  case CBO::BO_LT:
  case CBO::BO_LE:
    isIncrementing = true;
    return true;
  case CBO::BO_GT:
  case CBO::BO_GE:
    isIncrementing = false;
    return true;
  default:
    return false;
  }
}

mlir::Value extractStep(clang::Expr *incExpr, const clang::VarDecl *var,
                        bool isIncrementing, mlir::OpBuilder &builder,
                        mlir::Location loc,
                        std::function<mlir::Value(clang::Expr *)> genExpr) {

  if (auto *unary = mlir::dyn_cast_or_null<clang::UnaryOperator>(incExpr)) {
    auto *subj = mlir::dyn_cast<clang::DeclRefExpr>(
        unary->getSubExpr()->IgnoreImpCasts());
    if (!subj || subj->getDecl() != var)
      return {};

    bool isInc = (unary->getOpcode() == clang::UO_PostInc ||
                  unary->getOpcode() == clang::UO_PreInc);
    bool isDec = (unary->getOpcode() == clang::UO_PostDec ||
                  unary->getOpcode() == clang::UO_PreDec);

    if ((isInc && isIncrementing) || (isDec && !isIncrementing))
      return indexConst(builder, loc, 1);
    return {};
  }

  if (auto *bin = mlir::dyn_cast_or_null<clang::BinaryOperator>(incExpr)) {
    auto *lhsRef =
        mlir::dyn_cast<clang::DeclRefExpr>(bin->getLHS()->IgnoreImpCasts());
    if (!lhsRef || lhsRef->getDecl() != var)
      return {};

    auto op = bin->getOpcode();

    // i += n  /  i -= n
    if ((op == clang::BO_AddAssign && isIncrementing) ||
        (op == clang::BO_SubAssign && !isIncrementing)) {
      mlir::Value s = genExpr(bin->getRHS());
      return toIndex(builder, loc, s);
    }

    // i = i + n  /  i = i - n
    if (op == clang::BO_Assign) {
      auto *rhs = mlir::dyn_cast<clang::BinaryOperator>(
          bin->getRHS()->IgnoreImpCasts());
      if (!rhs)
        return {};
      auto *rhsLHS =
          mlir::dyn_cast<clang::DeclRefExpr>(rhs->getLHS()->IgnoreImpCasts());
      if (!rhsLHS || rhsLHS->getDecl() != var)
        return {};

      auto rhsOp = rhs->getOpcode();
      if ((rhsOp == clang::BO_Add && isIncrementing) ||
          (rhsOp == clang::BO_Sub && !isIncrementing)) {
        mlir::Value s = genExpr(rhs->getRHS());
        return toIndex(builder, loc, s);
      }
    }
  }

  return {};
}

void adjustBounds(mlir::OpBuilder &builder, mlir::Location loc,
                  clang::BinaryOperatorKind condOp, bool isIncrementing,
                  mlir::Value initVal, mlir::Value condVal, mlir::Value &lb,
                  mlir::Value &ub) {
  if (isIncrementing) {
    lb = initVal;
    ub = condVal;
    if (condOp == clang::BO_LE)
      ub = addi(builder, loc, ub, 1);
  } else {
    lb = condVal;
    ub = initVal;
    if (condOp == clang::BO_GT) {
      lb = addi(builder, loc, lb, 1);
      ub = addi(builder, loc, ub, 1);
    } else if (condOp == clang::BO_GE) {
      ub = addi(builder, loc, ub, 1);
    }
  }
}

std::optional<SimpleLoopInfo>
analyseForLoop(clang::ForStmt *forStmt, mlir::OpBuilder &builder,
               mlir::Location loc,
               std::function<mlir::Value(clang::Expr *)> genExpr) {

  auto *initStmt = mlir::dyn_cast_or_null<clang::DeclStmt>(forStmt->getInit());
  if (!initStmt || !initStmt->isSingleDecl())
    return std::nullopt;

  auto *varDecl = mlir::dyn_cast<clang::VarDecl>(initStmt->getSingleDecl());
  if (!varDecl || !varDecl->hasInit() || !varDecl->getType()->isIntegerType())
    return std::nullopt;

  auto *cond =
      mlir::dyn_cast_or_null<clang::BinaryOperator>(forStmt->getCond());
  if (!cond)
    return std::nullopt;

  auto *condLHSRef =
      mlir::dyn_cast<clang::DeclRefExpr>(cond->getLHS()->IgnoreImpCasts());
  if (!condLHSRef || condLHSRef->getDecl() != varDecl)
    return std::nullopt;

  bool isIncrementing = true;
  if (!classifyCondOp(cond->getOpcode(), isIncrementing))
    return std::nullopt;

  mlir::Value step = extractStep(forStmt->getInc(), varDecl, isIncrementing,
                                 builder, loc, genExpr);
  if (!step)
    return std::nullopt;

  mlir::Value initVal = toIndex(builder, loc, genExpr(varDecl->getInit()));
  mlir::Value condVal = toIndex(builder, loc, genExpr(cond->getRHS()));

  mlir::Value lb, ub;
  adjustBounds(builder, loc, cond->getOpcode(), isIncrementing, initVal,
               condVal, lb, ub);

  return SimpleLoopInfo{varDecl, lb, ub, step, isIncrementing};
}

} // namespace cmlirc::detail
