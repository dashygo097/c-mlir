#ifndef CMLIRC_CGUTILS_H
#define CMLIRC_CGUTILS_H

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"
#include "clang/AST/Expr.h"
#include "clang/AST/Stmt.h"

namespace cmlirc::detail {

struct SimpleLoopInfo {
  const clang::VarDecl *inductionVar = nullptr;
  mlir::Value lowerBound;
  mlir::Value upperBound;
  mlir::Value step;
  bool isIncrementing = true;

  explicit operator bool() const {
    return inductionVar && lowerBound && upperBound && step;
  }
};

inline mlir::Value indexConst(mlir::OpBuilder &b, mlir::Location loc,
                              int64_t v) {
  return mlir::arith::ConstantOp::create(b, loc, b.getIndexType(),
                                         b.getIndexAttr(v))
      .getResult();
}

inline mlir::Value intConst(mlir::OpBuilder &b, mlir::Location loc,
                            mlir::Type t, int64_t v) {
  return mlir::arith::ConstantOp::create(b, loc, t, b.getIntegerAttr(t, v))
      .getResult();
}

inline mlir::Value toIndex(mlir::OpBuilder &b, mlir::Location loc,
                           mlir::Value v) {
  if (v.getType().isIndex())
    return v;
  return mlir::arith::IndexCastOp::create(b, loc, b.getIndexType(), v)
      .getResult();
}

inline mlir::Value addOne(mlir::OpBuilder &b, mlir::Location loc,
                          mlir::Value v) {
  mlir::Value one = intConst(b, loc, v.getType(), 1);
  return mlir::arith::AddIOp::create(b, loc, v, one).getResult();
}

inline std::optional<int64_t> getConstantInt(mlir::Value v) {
  while (auto cast = v.getDefiningOp<mlir::arith::IndexCastOp>())
    v = cast.getIn();
  if (auto c = v.getDefiningOp<mlir::arith::ConstantOp>())
    if (auto ia = mlir::dyn_cast<mlir::IntegerAttr>(c.getValue()))
      return ia.getInt();
  return std::nullopt;
}

inline void ensureYield(mlir::OpBuilder &b, mlir::Location loc,
                        mlir::Block *block) {
  if (block->empty() ||
      !block->back().hasTrait<mlir::OpTrait::IsTerminator>()) {
    mlir::OpBuilder::InsertionGuard g(b);
    b.setInsertionPointToEnd(block);
    mlir::scf::YieldOp::create(b, loc, mlir::ValueRange{});
  }
}

static bool classifyCondOp(clang::BinaryOperatorKind op, bool &isIncrementing) {
  switch (op) {
  case clang::BO_LT:
  case clang::BO_LE:
    isIncrementing = true;
    return true;
  case clang::BO_GT:
  case clang::BO_GE:
    isIncrementing = false;
    return true;
  default:
    return false;
  }
}

static mlir::Value
extractStep(clang::Expr *incExpr, const clang::VarDecl *var,
            bool isIncrementing, mlir::OpBuilder &b, mlir::Location loc,
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
      return indexConst(b, loc, 1);
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
      return toIndex(b, loc, s);
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
        return toIndex(b, loc, s);
      }
    }
  }

  return {};
}

static void adjustBounds(mlir::OpBuilder &b, mlir::Location loc,
                         clang::BinaryOperatorKind condOp, bool isIncrementing,
                         mlir::Value initVal, mlir::Value condVal,
                         mlir::Value &lb, mlir::Value &ub) {
  if (isIncrementing) {
    lb = initVal;
    ub = condVal;
    if (condOp == clang::BO_LE)
      ub = addOne(b, loc, ub);
  } else {
    lb = condVal;
    ub = initVal;
    if (condOp == clang::BO_GT) {
      lb = addOne(b, loc, lb);
      ub = addOne(b, loc, ub);
    } else if (condOp == clang::BO_GE) {
      ub = addOne(b, loc, ub);
    }
  }
}

static std::optional<SimpleLoopInfo>
analyseForLoop(clang::ForStmt *forStmt, mlir::OpBuilder &b, mlir::Location loc,
               std::function<mlir::Value(clang::Expr *)> genExpr) {

  auto *initStmt = mlir::dyn_cast_or_null<clang::DeclStmt>(forStmt->getInit());
  if (!initStmt || !initStmt->isSingleDecl())
    return std::nullopt;

  auto *varDecl = mlir::dyn_cast<clang::VarDecl>(initStmt->getSingleDecl());
  if (!varDecl || !varDecl->hasInit())
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

  mlir::Value step =
      extractStep(forStmt->getInc(), varDecl, isIncrementing, b, loc, genExpr);
  if (!step)
    return std::nullopt;

  mlir::Value initVal = toIndex(b, loc, genExpr(varDecl->getInit()));
  mlir::Value condVal = toIndex(b, loc, genExpr(cond->getRHS()));

  mlir::Value lb, ub;
  adjustBounds(b, loc, cond->getOpcode(), isIncrementing, initVal, condVal, lb,
               ub);

  return SimpleLoopInfo{varDecl, lb, ub, step, isIncrementing};
}

} // namespace cmlirc::detail

#endif // CMLIRC_CGUTILS_H
