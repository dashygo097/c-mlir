#include "../../../Converter.h"
#include "../../Types/Types.h"
#include "mlir/Dialect/SCF/IR/SCF.h"

namespace cmlirc {

bool CMLIRConverter::TraverseForStmt(clang::ForStmt *forStmt) {
  if (!currentFunc) {
    return true;
  }

  mlir::OpBuilder &builder = context_manager_.Builder();
  mlir::Location loc = builder.getUnknownLoc();

  // Process initialization
  if (forStmt->getInit()) {
    TraverseStmt(forStmt->getInit());
  }

  bool isSimpleLoop = false;
  bool isIncrementing = true;
  const clang::VarDecl *inductionVar = nullptr;
  mlir::Value lowerBound, upperBound, step;

  if (auto *init =
          llvm::dyn_cast_or_null<clang::DeclStmt>(forStmt->getInit())) {
    if (init->isSingleDecl()) {
      if (auto *varDecl =
              llvm::dyn_cast<clang::VarDecl>(init->getSingleDecl())) {
        if (varDecl->hasInit()) {

          if (auto *cond = llvm::dyn_cast_or_null<clang::BinaryOperator>(
                  forStmt->getCond())) {

            auto *condLHS = cond->getLHS()->IgnoreImpCasts();
            if (auto *declRef = llvm::dyn_cast<clang::DeclRefExpr>(condLHS)) {
              if (declRef->getDecl() == varDecl) {

                clang::BinaryOperatorKind condOp = cond->getOpcode();

                bool validCondition = false;
                if (condOp == clang::BO_LT || condOp == clang::BO_LE) {
                  isIncrementing = true;
                  validCondition = true;
                } else if (condOp == clang::BO_GT || condOp == clang::BO_GE) {
                  isIncrementing = false;
                  validCondition = true;
                }

                if (validCondition) {
                  mlir::Value stepValue = nullptr;
                  bool validIncrement = false;

                  if (auto *inc = llvm::dyn_cast_or_null<clang::UnaryOperator>(
                          forStmt->getInc())) {
                    auto *incSubExpr = inc->getSubExpr()->IgnoreImpCasts();
                    if (auto *incVar =
                            llvm::dyn_cast<clang::DeclRefExpr>(incSubExpr)) {
                      if (incVar->getDecl() == varDecl) {
                        clang::UnaryOperatorKind incOp = inc->getOpcode();

                        if ((incOp == clang::UO_PostInc ||
                             incOp == clang::UO_PreInc) &&
                            isIncrementing) {
                          validIncrement = true;
                          stepValue = mlir::arith::ConstantOp::create(
                                          builder, loc, builder.getIndexType(),
                                          builder.getIndexAttr(1))
                                          .getResult();
                        } else if ((incOp == clang::UO_PostDec ||
                                    incOp == clang::UO_PreDec) &&
                                   !isIncrementing) {
                          validIncrement = true;
                          stepValue = mlir::arith::ConstantOp::create(
                                          builder, loc, builder.getIndexType(),
                                          builder.getIndexAttr(1))
                                          .getResult();
                        }
                      }
                    }
                  }

                  else if (auto *inc =
                               llvm::dyn_cast_or_null<clang::BinaryOperator>(
                                   forStmt->getInc())) {
                    auto *incLHS = inc->getLHS()->IgnoreImpCasts();
                    if (auto *incVar =
                            llvm::dyn_cast<clang::DeclRefExpr>(incLHS)) {
                      if (incVar->getDecl() == varDecl) {
                        clang::BinaryOperatorKind incOp = inc->getOpcode();

                        if (incOp == clang::BO_AddAssign && isIncrementing) {
                          validIncrement = true;
                          stepValue = generateExpr(inc->getRHS());
                        } else if (incOp == clang::BO_SubAssign &&
                                   !isIncrementing) {
                          validIncrement = true;
                          stepValue = generateExpr(inc->getRHS());
                        } else if (incOp == clang::BO_Assign) {
                          if (auto *rhs = llvm::dyn_cast<clang::BinaryOperator>(
                                  inc->getRHS()->IgnoreImpCasts())) {
                            auto *rhsLHS = rhs->getLHS()->IgnoreImpCasts();
                            if (auto *rhsVar =
                                    llvm::dyn_cast<clang::DeclRefExpr>(
                                        rhsLHS)) {
                              if (rhsVar->getDecl() == varDecl) {
                                if (rhs->getOpcode() == clang::BO_Add &&
                                    isIncrementing) {
                                  validIncrement = true;
                                  stepValue = generateExpr(rhs->getRHS());
                                } else if (rhs->getOpcode() == clang::BO_Sub &&
                                           !isIncrementing) {
                                  validIncrement = true;
                                  stepValue = generateExpr(rhs->getRHS());
                                }
                              }
                            }
                          }
                        }

                        if (validIncrement && stepValue &&
                            !stepValue.getType().isIndex()) {
                          stepValue = mlir::arith::IndexCastOp::create(
                                          builder, loc, builder.getIndexType(),
                                          stepValue)
                                          .getResult();
                        }
                      }
                    }
                  }

                  if (validIncrement && stepValue) {
                    isSimpleLoop = true;
                    inductionVar = varDecl;
                    step = stepValue;

                    mlir::Value initVal = generateExpr(varDecl->getInit());
                    mlir::Value condVal = generateExpr(cond->getRHS());

                    if (isIncrementing) {
                      lowerBound = initVal;
                      upperBound = condVal;

                      if (condOp == clang::BO_LE) {
                        mlir::Type ubType = upperBound.getType();
                        mlir::Value one = mlir::arith::ConstantOp::create(
                                              builder, loc, ubType,
                                              builder.getIntegerAttr(ubType, 1))
                                              .getResult();
                        upperBound = mlir::arith::AddIOp::create(
                                         builder, loc, upperBound, one)
                                         .getResult();
                      }
                    } else {
                      lowerBound = condVal;
                      upperBound = initVal;

                      if (condOp == clang::BO_GE) {
                        mlir::Type ubType = upperBound.getType();
                        mlir::Value one = mlir::arith::ConstantOp::create(
                                              builder, loc, ubType,
                                              builder.getIntegerAttr(ubType, 1))
                                              .getResult();
                        upperBound = mlir::arith::AddIOp::create(
                                         builder, loc, upperBound, one)
                                         .getResult();
                      } else if (condOp == clang::BO_GT) {
                        mlir::Type lbType = lowerBound.getType();
                        mlir::Value one = mlir::arith::ConstantOp::create(
                                              builder, loc, lbType,
                                              builder.getIntegerAttr(lbType, 1))
                                              .getResult();
                        lowerBound = mlir::arith::AddIOp::create(
                                         builder, loc, lowerBound, one)
                                         .getResult();

                        mlir::Type ubType = upperBound.getType();
                        one = mlir::arith::ConstantOp::create(
                                  builder, loc, ubType,
                                  builder.getIntegerAttr(ubType, 1))
                                  .getResult();
                        upperBound = mlir::arith::AddIOp::create(
                                         builder, loc, upperBound, one)
                                         .getResult();
                      }
                    }
                  }
                }
              }
            }
          }
        }
      }
    }
  }

  if (isSimpleLoop && lowerBound && upperBound && step && inductionVar) {
    if (!lowerBound.getType().isIndex()) {
      lowerBound = mlir::arith::IndexCastOp::create(
                       builder, loc, builder.getIndexType(), lowerBound)
                       .getResult();
    }
    if (!upperBound.getType().isIndex()) {
      upperBound = mlir::arith::IndexCastOp::create(
                       builder, loc, builder.getIndexType(), upperBound)
                       .getResult();
    }

    auto forOp =
        mlir::scf::ForOp::create(builder, loc, lowerBound, upperBound, step);

    builder.setInsertionPointToStart(forOp.getBody());

    mlir::Value inductionValue = forOp.getInductionVar();
    mlir::Type origType = convertType(builder, inductionVar->getType());

    if (!isIncrementing) {
      mlir::Value one =
          mlir::arith::ConstantOp::create(builder, loc, builder.getIndexType(),
                                          builder.getIndexAttr(1))
              .getResult();

      mlir::Value adjustedUpper =
          mlir::arith::SubIOp::create(builder, loc, upperBound, one)
              .getResult();

      mlir::Value offset =
          mlir::arith::SubIOp::create(builder, loc, inductionValue, lowerBound)
              .getResult();

      inductionValue =
          mlir::arith::SubIOp::create(builder, loc, adjustedUpper, offset)
              .getResult();
    }

    if (!origType.isIndex()) {
      inductionValue = mlir::arith::IndexCastOp::create(builder, loc, origType,
                                                        inductionValue)
                           .getResult();
    }

    if (symbolTable.count(inductionVar)) {
      mlir::Value memref = symbolTable[inductionVar];
      mlir::memref::StoreOp::create(builder, loc, inductionValue, memref,
                                    mlir::ValueRange{});
    }

    loopStack_.push_back({forOp.getBody(), nullptr});
    TraverseStmt(forStmt->getBody());
    loopStack_.pop_back();

    builder.setInsertionPointToEnd(forOp.getBody());
    if (forOp.getBody()->empty() ||
        !forOp.getBody()->back().hasTrait<mlir::OpTrait::IsTerminator>()) {
      mlir::scf::YieldOp::create(builder, builder.getUnknownLoc());
    }

    builder.setInsertionPointAfter(forOp);

  } else {
    auto whileOp = mlir::scf::WhileOp::create(builder, loc, mlir::TypeRange{},
                                              mlir::ValueRange{});

    mlir::Block *beforeBlock = &whileOp.getBefore().front();
    builder.setInsertionPointToStart(beforeBlock);

    mlir::Value condition;
    if (forStmt->getCond()) {
      condition = generateExpr(forStmt->getCond());
      condition = convertToBool(builder, condition);
    } else {
      condition =
          mlir::arith::ConstantOp::create(builder, loc, builder.getI1Type(),
                                          builder.getBoolAttr(true))
              .getResult();
    }

    mlir::scf::ConditionOp::create(builder, loc, condition, mlir::ValueRange{});

    mlir::Block *afterBlock = &whileOp.getAfter().front();
    builder.setInsertionPointToStart(afterBlock);

    loopStack_.push_back({beforeBlock, afterBlock});

    TraverseStmt(forStmt->getBody());

    if (forStmt->getInc()) {
      generateExpr(forStmt->getInc());
    }

    loopStack_.pop_back();

    builder.setInsertionPointToEnd(afterBlock);
    if (afterBlock->empty() ||
        !afterBlock->back().hasTrait<mlir::OpTrait::IsTerminator>()) {
      mlir::scf::YieldOp::create(builder, loc, mlir::ValueRange{});
    }

    builder.setInsertionPointAfter(whileOp);
  }

  return true;
}

} // namespace cmlirc
