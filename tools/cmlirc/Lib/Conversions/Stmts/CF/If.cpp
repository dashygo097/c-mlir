#include "../../../Converter.h"
#include "../../Types/Types.h"
#include "mlir/Dialect/SCF/IR/SCF.h"

namespace cmlirc {

bool CMLIRConverter::TraverseIfStmt(clang::IfStmt *ifStmt) {
  if (!currentFunc) {
    return true;
  }

  mlir::OpBuilder &builder = context_manager_.Builder();
  mlir::Location loc = builder.getUnknownLoc();

  mlir::Value condition = generateExpr(ifStmt->getCond());
  if (!condition) {
    llvm::errs() << "Failed to generate if condition\n";
    return false;
  }

  mlir::Value condBool = convertToBool(builder, condition);

  bool hasElse = ifStmt->getElse() != nullptr;

  bool thenHasReturn = branchEndsWithReturn(ifStmt->getThen());
  bool elseHasReturn = hasElse && branchEndsWithReturn(ifStmt->getElse());
  bool bothReturn = thenHasReturn && elseHasReturn;

  // Pattern 1: if (cond) return a; else return b;
  if (bothReturn && hasElse &&
      currentFunc.getFunctionType().getNumResults() > 0) {
    clang::Expr *thenReturnExpr = nullptr;
    clang::Expr *elseReturnExpr = nullptr;

    auto extractReturnExpr = [](clang::Stmt *stmt) -> clang::Expr * {
      if (auto *retStmt = llvm::dyn_cast<clang::ReturnStmt>(stmt)) {
        return retStmt->getRetValue();
      }
      if (auto *compound = llvm::dyn_cast<clang::CompoundStmt>(stmt)) {
        if (compound->size() == 1) {
          if (auto *ret =
                  llvm::dyn_cast<clang::ReturnStmt>(compound->body_back())) {
            return ret->getRetValue();
          }
        }
      }
      return nullptr;
    };

    thenReturnExpr = extractReturnExpr(ifStmt->getThen());
    elseReturnExpr = extractReturnExpr(ifStmt->getElse());

    if (thenReturnExpr && elseReturnExpr) {
      mlir::Value thenValue = generateExpr(thenReturnExpr);
      mlir::Value elseValue = generateExpr(elseReturnExpr);

      if (thenValue && elseValue &&
          thenValue.getType() == elseValue.getType()) {
        mlir::Value selectResult =
            mlir::arith::SelectOp::create(builder, loc, condBool, thenValue,
                                          elseValue)
                .getResult();

        if (returnValueCapture) {
          *returnValueCapture = selectResult;
        } else {
          mlir::func::ReturnOp::create(builder, loc, selectResult);
        }
        return true;
      }
    }
  }

  // Pattern 2: if (cond) { assignments... } else { assignments... }
  if (!bothReturn && hasElse) {
    struct Assignment {
      const clang::VarDecl *var;
      clang::Expr *expr;
    };

    auto collectAssignments =
        [](clang::Stmt *stmt,
           llvm::SmallVectorImpl<Assignment> &assignments) -> bool {
      llvm::SmallVector<clang::Stmt *, 8> stmts;

      if (auto *compound = llvm::dyn_cast<clang::CompoundStmt>(stmt)) {
        for (auto *s : compound->body()) {
          stmts.push_back(s);
        }
      } else {
        stmts.push_back(stmt);
      }

      for (auto *s : stmts) {
        if (auto *binOp = llvm::dyn_cast<clang::BinaryOperator>(s)) {
          if (binOp->getOpcode() == clang::BO_Assign) {
            if (auto *declRef = llvm::dyn_cast<clang::DeclRefExpr>(
                    binOp->getLHS()->IgnoreImpCasts())) {
              if (auto *varDecl =
                      llvm::dyn_cast<clang::VarDecl>(declRef->getDecl())) {
                assignments.push_back({varDecl, binOp->getRHS()});
                continue;
              }
            }
          }
        }
        return false;
      }
      return true;
    };

    llvm::SmallVector<Assignment, 4> thenAssignments;
    llvm::SmallVector<Assignment, 4> elseAssignments;

    bool thenOnlyAssignments =
        collectAssignments(ifStmt->getThen(), thenAssignments);
    bool elseOnlyAssignments =
        collectAssignments(ifStmt->getElse(), elseAssignments);

    if (thenOnlyAssignments && elseOnlyAssignments &&
        !thenAssignments.empty() &&
        thenAssignments.size() == elseAssignments.size()) {

      bool sameVariables = true;
      for (size_t i = 0; i < thenAssignments.size(); ++i) {
        if (thenAssignments[i].var != elseAssignments[i].var) {
          sameVariables = false;
          break;
        }
      }

      if (sameVariables) {
        for (size_t i = 0; i < thenAssignments.size(); ++i) {
          mlir::Value thenValue = generateExpr(thenAssignments[i].expr);
          mlir::Value elseValue = generateExpr(elseAssignments[i].expr);

          if (thenValue && elseValue &&
              thenValue.getType() == elseValue.getType()) {
            mlir::Value selectResult =
                mlir::arith::SelectOp::create(builder, loc, condBool, thenValue,
                                              elseValue)
                    .getResult();

            const clang::VarDecl *targetVar = thenAssignments[i].var;
            if (symbolTable.count(targetVar)) {
              mlir::Value memref = symbolTable[targetVar];
              mlir::memref::StoreOp::create(builder, loc, selectResult, memref,
                                            mlir::ValueRange{});
            }
          } else {
            sameVariables = false;
            break;
          }
        }

        if (sameVariables) {
          return true;
        }
      }
    }
  }

  bool isNested = (returnValueCapture != nullptr);

  llvm::SmallVector<mlir::Type, 1> resultTypes;
  if (bothReturn && currentFunc.getFunctionType().getNumResults() > 0) {
    resultTypes.push_back(currentFunc.getFunctionType().getResult(0));
  }

  auto ifOp = mlir::scf::IfOp::create(
      builder, loc, mlir::TypeRange{resultTypes}, condBool, hasElse);

  mlir::Block *thenBlock = &ifOp.getThenRegion().front();
  builder.setInsertionPointToStart(thenBlock);

  mlir::Value thenReturnValue = nullptr;
  mlir::Value *savedReturnCapture = returnValueCapture;
  if (bothReturn) {
    returnValueCapture = &thenReturnValue;
  }

  TraverseStmt(ifStmt->getThen());

  returnValueCapture = savedReturnCapture;

  builder.setInsertionPointToEnd(thenBlock);

  if (thenBlock->empty() ||
      !thenBlock->back().hasTrait<mlir::OpTrait::IsTerminator>()) {
    if (bothReturn && thenReturnValue) {
      mlir::scf::YieldOp::create(builder, loc, thenReturnValue);
    } else {
      mlir::scf::YieldOp::create(builder, builder.getUnknownLoc());
    }
  } else if (bothReturn && llvm::isa<mlir::func::ReturnOp>(thenBlock->back())) {
    auto returnOp = llvm::cast<mlir::func::ReturnOp>(thenBlock->back());
    mlir::ValueRange returnOperands = returnOp.getOperands();
    returnOp.erase();
    if (!returnOperands.empty()) {
      mlir::scf::YieldOp::create(builder, loc, returnOperands[0]);
    } else {
      mlir::scf::YieldOp::create(builder, builder.getUnknownLoc());
    }
  }

  if (hasElse) {
    mlir::Block *elseBlock = &ifOp.getElseRegion().front();
    builder.setInsertionPointToStart(elseBlock);

    mlir::Value elseReturnValue = nullptr;
    savedReturnCapture = returnValueCapture;
    if (bothReturn) {
      returnValueCapture = &elseReturnValue;
    }

    TraverseStmt(ifStmt->getElse());

    returnValueCapture = savedReturnCapture;

    builder.setInsertionPointToEnd(elseBlock);

    if (elseBlock->empty() ||
        !elseBlock->back().hasTrait<mlir::OpTrait::IsTerminator>()) {
      if (bothReturn && elseReturnValue) {
        mlir::scf::YieldOp::create(builder, loc, elseReturnValue);
      } else {
        mlir::scf::YieldOp::create(builder, builder.getUnknownLoc());
      }
    } else if (bothReturn &&
               llvm::isa<mlir::func::ReturnOp>(elseBlock->back())) {
      auto returnOp = llvm::cast<mlir::func::ReturnOp>(elseBlock->back());
      mlir::ValueRange returnOperands = returnOp.getOperands();
      returnOp.erase();
      if (!returnOperands.empty()) {
        mlir::scf::YieldOp::create(builder, loc, returnOperands[0]);
      } else {
        mlir::scf::YieldOp::create(builder, builder.getUnknownLoc());
      }
    }
  }

  builder.setInsertionPointAfter(ifOp);

  if (bothReturn && ifOp.getNumResults() > 0) {
    if (isNested && savedReturnCapture) {
      *savedReturnCapture = ifOp.getResult(0);
    } else {
      mlir::func::ReturnOp::create(builder, loc, ifOp.getResult(0));
    }
  }

  return true;
}

} // namespace cmlirc
