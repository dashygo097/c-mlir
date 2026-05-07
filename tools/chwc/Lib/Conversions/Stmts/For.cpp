#include "../../Converter.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/WithColor.h"

namespace chwc {

auto stripExpr(clang::Expr *expr) -> clang::Expr * {
  if (!expr) {
    return nullptr;
  }

  return expr->IgnoreParenImpCasts();
}

auto getDeclRefVar(clang::Expr *expr) -> const clang::VarDecl * {
  expr = stripExpr(expr);

  auto *declRef = llvm::dyn_cast_or_null<clang::DeclRefExpr>(expr);
  if (!declRef) {
    return nullptr;
  }

  return llvm::dyn_cast<clang::VarDecl>(declRef->getDecl());
}

auto getIntegerLiteralValue(clang::Expr *expr, int64_t &value) -> bool {
  expr = stripExpr(expr);

  if (auto *intLit = llvm::dyn_cast_or_null<clang::IntegerLiteral>(expr)) {
    value = intLit->getValue().getSExtValue();
    return true;
  }

  if (auto *boolLit = llvm::dyn_cast_or_null<clang::CXXBoolLiteralExpr>(expr)) {
    value = boolLit->getValue() ? 1 : 0;
    return true;
  }

  return false;
}

auto parseForInit(clang::Stmt *init, const clang::VarDecl *&varDecl,
                  int64_t &initialValue) -> bool {
  auto *declStmt = llvm::dyn_cast_or_null<clang::DeclStmt>(init);
  if (!declStmt || !declStmt->isSingleDecl()) {
    return false;
  }

  auto *decl = llvm::dyn_cast<clang::VarDecl>(declStmt->getSingleDecl());
  if (!decl || !decl->getInit()) {
    return false;
  }

  int64_t value = 0;
  if (!getIntegerLiteralValue(decl->getInit(), value)) {
    return false;
  }

  varDecl = decl;
  initialValue = value;
  return true;
}

auto parseForCond(clang::Expr *cond, const clang::VarDecl *varDecl,
                  clang::BinaryOperatorKind &op, int64_t &bound) -> bool {
  auto *binOp = llvm::dyn_cast_or_null<clang::BinaryOperator>(stripExpr(cond));
  if (!binOp) {
    return false;
  }

  const clang::VarDecl *lhsVar = getDeclRefVar(binOp->getLHS());
  if (!lhsVar || lhsVar->getCanonicalDecl() != varDecl->getCanonicalDecl()) {
    return false;
  }

  int64_t rhsValue = 0;
  if (!getIntegerLiteralValue(binOp->getRHS(), rhsValue)) {
    return false;
  }

  switch (binOp->getOpcode()) {
  case clang::BO_LT:
  case clang::BO_LE:
  case clang::BO_GT:
  case clang::BO_GE:
  case clang::BO_NE:
    op = binOp->getOpcode();
    bound = rhsValue;
    return true;

  default:
    return false;
  }
}

auto parseForInc(clang::Stmt *inc, const clang::VarDecl *varDecl, int64_t &step)
    -> bool {
  auto *unOp = llvm::dyn_cast_or_null<clang::UnaryOperator>(inc);
  if (unOp) {
    const clang::VarDecl *target = getDeclRefVar(unOp->getSubExpr());
    if (!target || target->getCanonicalDecl() != varDecl->getCanonicalDecl()) {
      return false;
    }

    if (unOp->getOpcode() == clang::UO_PreInc ||
        unOp->getOpcode() == clang::UO_PostInc) {
      step = 1;
      return true;
    }

    if (unOp->getOpcode() == clang::UO_PreDec ||
        unOp->getOpcode() == clang::UO_PostDec) {
      step = -1;
      return true;
    }

    return false;
  }

  auto *binOp = llvm::dyn_cast_or_null<clang::BinaryOperator>(inc);
  if (!binOp) {
    return false;
  }

  const clang::VarDecl *lhsVar = getDeclRefVar(binOp->getLHS());
  if (!lhsVar || lhsVar->getCanonicalDecl() != varDecl->getCanonicalDecl()) {
    return false;
  }

  int64_t rhsValue = 0;

  if (binOp->getOpcode() == clang::BO_AddAssign &&
      getIntegerLiteralValue(binOp->getRHS(), rhsValue)) {
    step = rhsValue;
    return step != 0;
  }

  if (binOp->getOpcode() == clang::BO_SubAssign &&
      getIntegerLiteralValue(binOp->getRHS(), rhsValue)) {
    step = -rhsValue;
    return step != 0;
  }

  return false;
}

auto conditionHolds(int64_t value, clang::BinaryOperatorKind op, int64_t bound)
    -> bool {
  switch (op) {
  case clang::BO_LT:
    return value < bound;
  case clang::BO_LE:
    return value <= bound;
  case clang::BO_GT:
    return value > bound;
  case clang::BO_GE:
    return value >= bound;
  case clang::BO_NE:
    return value != bound;
  default:
    return false;
  }
}

auto CHWConverter::TraverseForStmt(clang::ForStmt *forStmt) -> bool {
  if (!forStmt) {
    return true;
  }

  const clang::VarDecl *loopVar = nullptr;
  int64_t initialValue = 0;
  clang::BinaryOperatorKind condOp = clang::BO_LT;
  int64_t bound = 0;
  int64_t step = 0;

  if (!parseForInit(forStmt->getInit(), loopVar, initialValue) ||
      !parseForCond(forStmt->getCond(), loopVar, condOp, bound) ||
      !parseForInc(forStmt->getInc(), loopVar, step)) {
    llvm::WithColor::error()
        << "chwc: only static canonical for-loops are supported\n";
    return true;
  }

  mlir::OpBuilder &builder = contextManager.Builder();
  mlir::Location loc = builder.getUnknownLoc();

  mlir::Type loopType = convertType(loopVar->getType());
  auto intType = mlir::dyn_cast_or_null<mlir::IntegerType>(loopType);
  if (!intType) {
    llvm::WithColor::error()
        << "chwc: static for-loop variable must lower to integer type\n";
    return true;
  }

  auto oldValueIt = localValueTable.find(loopVar);
  bool hadOldValue = oldValueIt != localValueTable.end();
  mlir::Value oldValue = hadOldValue ? oldValueIt->second : mlir::Value{};

  auto oldConstIt = localConstIntTable.find(loopVar);
  bool hadOldConst = oldConstIt != localConstIntTable.end();
  int64_t oldConst = hadOldConst ? oldConstIt->second : 0;

  constexpr uint64_t maxStaticIterations = 1048576;
  uint64_t iterations = 0;

  for (int64_t value = initialValue; conditionHolds(value, condOp, bound);
       value += step) {
    if (++iterations > maxStaticIterations) {
      llvm::WithColor::error()
          << "chwc: static for-loop iteration limit exceeded\n";
      break;
    }

    mlir::Value mlirValue = mlir::arith::ConstantIntOp::create(
                                builder, loc, value, intType.getWidth())
                                .getResult();

    localValueTable[loopVar] = mlirValue;
    localConstIntTable[loopVar] = value;

    TraverseStmt(forStmt->getBody());
  }

  if (hadOldValue) {
    localValueTable[loopVar] = oldValue;
  } else {
    localValueTable.erase(loopVar);
  }

  if (hadOldConst) {
    localConstIntTable[loopVar] = oldConst;
  } else {
    localConstIntTable.erase(loopVar);
  }

  return true;
}

} // namespace chwc
