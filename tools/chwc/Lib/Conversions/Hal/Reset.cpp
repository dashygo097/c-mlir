#include "../../Converter.h"
#include "../Utils/Cast.h"
#include "../Utils/Constant.h"
#include "clang/AST/ExprCXX.h"
#include "llvm/Support/WithColor.h"

namespace chwc {

auto getResetAssign(clang::Stmt *stmt, clang::Expr *&lhs, clang::Expr *&rhs)
    -> bool {
  if (auto *assignOp = mlir::dyn_cast<clang::BinaryOperator>(stmt)) {
    if (!assignOp->isAssignmentOp()) {
      return false;
    }

    lhs = assignOp->getLHS();
    rhs = assignOp->getRHS();
    return true;
  }

  if (auto *operatorCall = mlir::dyn_cast<clang::CXXOperatorCallExpr>(stmt)) {
    if (operatorCall->getOperator() != clang::OO_Equal ||
        operatorCall->getNumArgs() != 2) {
      return false;
    }

    lhs = operatorCall->getArg(0);
    rhs = operatorCall->getArg(1);
    return true;
  }

  if (auto *cleanups = mlir::dyn_cast<clang::ExprWithCleanups>(stmt)) {
    return getResetAssign(cleanups->getSubExpr(), lhs, rhs);
  }

  return false;
}

void CHWConverter::collectResetValues() {
  mlir::OpBuilder &builder = contextManager.Builder();
  mlir::Location loc = builder.getUnknownLoc();

  for (auto &[fieldDecl, fieldInfo] : fieldTable) {
    if (fieldInfo.kind == HWFieldKind::Reg) {
      fieldInfo.resetValue = utils::zeroValue(builder, loc, fieldInfo.type);
    }
  }

  for (clang::CXXMethodDecl *resetMethod : resetMethods) {
    if (!resetMethod || !resetMethod->hasBody()) {
      continue;
    }

    auto *body = mlir::dyn_cast<clang::CompoundStmt>(resetMethod->getBody());
    if (!body) {
      llvm::WithColor::error()
          << "chwc: reset method body must be compound stmt\n";
      continue;
    }

    for (clang::Stmt *stmt : body->body()) {
      clang::Expr *lhs = nullptr;
      clang::Expr *rhs = nullptr;

      if (!getResetAssign(stmt, lhs, rhs)) {
        llvm::WithColor::error()
            << "chwc: reset method only supports field = constant\n";
        continue;
      }

      const clang::FieldDecl *fieldDecl = getAssignedField(lhs);
      if (!fieldDecl || !fieldTable.count(fieldDecl)) {
        llvm::WithColor::error()
            << "chwc: reset assignment lhs must be hardware field\n";
        continue;
      }

      mlir::Value value = generateExpr(rhs);
      if (!value) {
        continue;
      }

      HWFieldInfo &fieldInfo = fieldTable[fieldDecl];
      value = utils::promoteValue(builder, loc, value, fieldInfo.type);
      if (!value) {
        continue;
      }

      fieldInfo.resetValue = value;
    }
  }
}

} // namespace chwc
