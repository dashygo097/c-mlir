#include "./ASTVisitor.h"
#include "../ArgumentList.h"
#include "./Conversions/Types.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "clang/Basic/SourceManager.h"

namespace cmlirc {
using namespace clang;

CMLIRCASTVisitor::CMLIRCASTVisitor(ContextManager &ctx)
    : context_manager_(ctx) {}

bool CMLIRCASTVisitor::TraverseFunctionDecl(clang::FunctionDecl *decl) {
  if (decl->isImplicit() || !decl->hasBody()) {
    return true;
  }

  mlir::OpBuilder &builder = context_manager_.Builder();
  builder.setInsertionPointToEnd(context_manager_.Module().getBody());

  // Convert parameter types
  llvm::SmallVector<mlir::Type, 4> argTypes;
  for (auto *param : decl->parameters()) {
    mlir::Type paramType = convertType(builder, param->getType());
    argTypes.push_back(paramType);
  }

  // Convert return type
  mlir::Type returnType = convertType(builder, decl->getReturnType());

  llvm::SmallVector<mlir::Type, 1> returnTypes;
  if (!mlir::isa<mlir::NoneType>(returnType)) {
    returnTypes.push_back(returnType);
  }

  auto funcType = builder.getFunctionType(argTypes, {returnType});

  // Create function
  auto funcOp = mlir::func::FuncOp::create(builder, builder.getUnknownLoc(),
                                           decl->getNameAsString(), funcType);

  // Create entry block
  auto *entryBlock = funcOp.addEntryBlock();
  builder.setInsertionPointToStart(entryBlock);

  currentFunc = funcOp;

  // Map parameters to block arguments
  for (unsigned i = 0; i < decl->getNumParams(); ++i) {
    auto *param = decl->getParamDecl(i);
    paramTable[param] = entryBlock->getArgument(i);
  }

  // Traverse function body manually
  TraverseStmt(decl->getBody());

  // Return true, but we've already handled traversal manually
  return true;
}

bool CMLIRCASTVisitor::TraverseVarDecl(VarDecl *decl) {
  if (decl->isImplicit()) {
    return true;
  }

  if (llvm::isa<clang::ParmVarDecl>(decl)) {
    return true;
  }

  if (!currentFunc) {
    return true;
  }

  SourceManager &SM = context_manager_.ClangContext().getSourceManager();
  SourceLocation loc = decl->getLocation();

  auto mlirLoc = mlir::FileLineColLoc::get(
      &context_manager_.MLIRContext(), SM.getFilename(loc),
      SM.getSpellingLineNumber(loc), SM.getSpellingColumnNumber(loc));

  mlir::OpBuilder &builder = context_manager_.Builder();

  QualType clangType = decl->getType();
  mlir::Type mlirType = convertType(builder, clangType);

  mlir::Type allocaType;
  if (auto memrefType = mlir::dyn_cast<mlir::MemRefType>(mlirType)) {
    allocaType = memrefType;
  } else {
    allocaType = mlir::MemRefType::get({}, mlirType);
  }

  // Create alloca
  auto allocaOp = mlir::memref::AllocaOp::create(
      builder, mlirLoc, mlir::dyn_cast<mlir::MemRefType>(allocaType));

  symbolTable[decl] = allocaOp.getResult();

  if (decl->hasInit()) {
    Expr *init = decl->getInit();
    mlir::Value initValue = generateExpr(init);

    if (initValue) {
      mlir::memref::StoreOp::create(builder, mlirLoc, initValue,
                                    allocaOp.getResult(), mlir::ValueRange{});
    }
  }

  return true;
}

bool CMLIRCASTVisitor::TraverseStmt(clang::Stmt *stmt) {
  if (!stmt || !currentFunc) {
    return RecursiveASTVisitor::TraverseStmt(stmt);
  }

  if (auto *expr = llvm::dyn_cast<clang::Expr>(stmt)) {
    if (hasSideEffects(expr)) {
      generateExpr(expr);
      return true;
    }
  }

  return RecursiveASTVisitor::TraverseStmt(stmt);
}

bool CMLIRCASTVisitor::TraverseReturnStmt(clang::ReturnStmt *stmt) {
  if (!currentFunc) {
    return true;
  }

  mlir::OpBuilder &builder = context_manager_.Builder();

  mlir::Value retValue = nullptr;
  if (auto *retExpr = stmt->getRetValue()) {
    retValue = generateExpr(retExpr);
  }

  if (retValue) {
    mlir::func::ReturnOp::create(builder, builder.getUnknownLoc(),
                                 mlir::ValueRange{retValue});
  } else {
    mlir::func::ReturnOp::create(builder, builder.getUnknownLoc());
  }

  return true;
}

bool CMLIRCASTVisitor::hasSideEffects(clang::Expr *expr) const {
  if (auto *unOp = llvm::dyn_cast<clang::UnaryOperator>(expr)) {
    return unOp->isIncrementDecrementOp();
  }

  if (auto *binOp = llvm::dyn_cast<clang::BinaryOperator>(expr)) {
    return binOp->isAssignmentOp() || binOp->isCompoundAssignmentOp();
  }

  if (llvm::isa<clang::CallExpr>(expr)) {
    return true;
  }

  return false;
}

bool CMLIRCASTVisitor::TraverseIfStmt(clang::IfStmt *ifStmt) {
  if (!currentFunc) {
    return true;
  }

  mlir::OpBuilder &builder = context_manager_.Builder();

  mlir::Value condition = generateExpr(ifStmt->getCond());
  if (!condition) {
    llvm::errs() << "Failed to generate if condition\n";
    return false;
  }

  mlir::Value condBool = convertToBool(condition);

  bool hasElse = ifStmt->getElse() != nullptr;

  auto ifOp = mlir::scf::IfOp::create(builder, builder.getUnknownLoc(),
                                      mlir::TypeRange{}, condBool, hasElse);

  mlir::Block *thenBlock = &ifOp.getThenRegion().front();
  builder.setInsertionPointToStart(thenBlock);

  TraverseStmt(ifStmt->getThen());

  builder.setInsertionPointToEnd(thenBlock);

  if (thenBlock->empty() ||
      !thenBlock->back().hasTrait<mlir::OpTrait::IsTerminator>()) {
    mlir::scf::YieldOp::create(builder, builder.getUnknownLoc());
  }

  if (hasElse) {
    mlir::Block *elseBlock = &ifOp.getElseRegion().front();
    builder.setInsertionPointToStart(elseBlock);

    TraverseStmt(ifStmt->getElse());

    builder.setInsertionPointToEnd(elseBlock);

    if (elseBlock->empty() ||
        !elseBlock->back().hasTrait<mlir::OpTrait::IsTerminator>()) {
      mlir::scf::YieldOp::create(builder, builder.getUnknownLoc());
    }
  }

  builder.setInsertionPointAfter(ifOp);

  return true;
}

bool CMLIRCASTVisitor::TraverseForStmt(clang::ForStmt *forStmt) {
  if (!currentFunc) {
    return true;
  }

  mlir::OpBuilder &builder = context_manager_.Builder();

  // Process initialization
  if (forStmt->getInit()) {
    TraverseStmt(forStmt->getInit());
  }

  // Analyze loop to determine if it's a simple counting loop
  bool isSimpleLoop = false;
  bool isIncrementing = true;
  const clang::VarDecl *inductionVar = nullptr;
  mlir::Value lowerBound, upperBound, step;

  // Check if init is a single variable declaration
  if (auto *init =
          llvm::dyn_cast_or_null<clang::DeclStmt>(forStmt->getInit())) {
    if (init->isSingleDecl()) {
      if (auto *varDecl =
              llvm::dyn_cast<clang::VarDecl>(init->getSingleDecl())) {
        if (varDecl->hasInit()) {

          // Check condition is a binary comparison
          if (auto *cond = llvm::dyn_cast_or_null<clang::BinaryOperator>(
                  forStmt->getCond())) {

            // Verify condition LHS references the induction variable
            auto *condLHS = cond->getLHS()->IgnoreImpCasts();
            if (auto *declRef = llvm::dyn_cast<clang::DeclRefExpr>(condLHS)) {
              if (declRef->getDecl() == varDecl) {

                clang::BinaryOperatorKind condOp = cond->getOpcode();

                // Determine loop direction from condition operator
                bool validCondition = false;
                if (condOp == clang::BO_LT || condOp == clang::BO_LE) {
                  isIncrementing = true;
                  validCondition = true;
                } else if (condOp == clang::BO_GT || condOp == clang::BO_GE) {
                  isIncrementing = false;
                  validCondition = true;
                }

                if (validCondition) {
                  // Analyze increment/decrement expression
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
                                          builder, builder.getUnknownLoc(),
                                          builder.getIndexType(),
                                          builder.getIndexAttr(1))
                                          .getResult();
                        } else if ((incOp == clang::UO_PostDec ||
                                    incOp == clang::UO_PreDec) &&
                                   !isIncrementing) {
                          validIncrement = true;
                          stepValue = mlir::arith::ConstantOp::create(
                                          builder, builder.getUnknownLoc(),
                                          builder.getIndexType(),
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

                        // Convert step to index type if needed
                        if (validIncrement && stepValue &&
                            !stepValue.getType().isIndex()) {
                          stepValue = mlir::arith::IndexCastOp::create(
                                          builder, builder.getUnknownLoc(),
                                          builder.getIndexType(), stepValue)
                                          .getResult();
                        }
                      }
                    }
                  }

                  if (validIncrement && stepValue) {
                    isSimpleLoop = true;
                    inductionVar = varDecl;
                    step = stepValue;

                    // Extract bounds from init and condition
                    mlir::Value initVal = generateExpr(varDecl->getInit());
                    mlir::Value condVal = generateExpr(cond->getRHS());

                    if (isIncrementing) {
                      lowerBound = initVal;
                      upperBound = condVal;

                      if (condOp == clang::BO_LE) {
                        mlir::Type ubType = upperBound.getType();
                        mlir::Value one =
                            mlir::arith::ConstantOp::create(
                                builder, builder.getUnknownLoc(), ubType,
                                builder.getIntegerAttr(ubType, 1))
                                .getResult();
                        upperBound = mlir::arith::AddIOp::create(
                                         builder, builder.getUnknownLoc(),
                                         upperBound, one)
                                         .getResult();
                      }
                    } else {
                      lowerBound = condVal;
                      upperBound = initVal;

                      if (condOp == clang::BO_GE) {
                        mlir::Type ubType = upperBound.getType();
                        mlir::Value one =
                            mlir::arith::ConstantOp::create(
                                builder, builder.getUnknownLoc(), ubType,
                                builder.getIntegerAttr(ubType, 1))
                                .getResult();
                        upperBound = mlir::arith::AddIOp::create(
                                         builder, builder.getUnknownLoc(),
                                         upperBound, one)
                                         .getResult();
                      } else if (condOp == clang::BO_GT) {
                        // i > end: lowerBound = end + 1, upperBound = start + 1
                        mlir::Type lbType = lowerBound.getType();
                        mlir::Value one =
                            mlir::arith::ConstantOp::create(
                                builder, builder.getUnknownLoc(), lbType,
                                builder.getIntegerAttr(lbType, 1))
                                .getResult();
                        lowerBound = mlir::arith::AddIOp::create(
                                         builder, builder.getUnknownLoc(),
                                         lowerBound, one)
                                         .getResult();

                        mlir::Type ubType = upperBound.getType();
                        one = mlir::arith::ConstantOp::create(
                                  builder, builder.getUnknownLoc(), ubType,
                                  builder.getIntegerAttr(ubType, 1))
                                  .getResult();
                        upperBound = mlir::arith::AddIOp::create(
                                         builder, builder.getUnknownLoc(),
                                         upperBound, one)
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

  // Generate code based on analysis
  if (isSimpleLoop && lowerBound && upperBound && step && inductionVar) {
    // Convert bounds to index type
    if (!lowerBound.getType().isIndex()) {
      lowerBound =
          mlir::arith::IndexCastOp::create(builder, builder.getUnknownLoc(),
                                           builder.getIndexType(), lowerBound)
              .getResult();
    }
    if (!upperBound.getType().isIndex()) {
      upperBound =
          mlir::arith::IndexCastOp::create(builder, builder.getUnknownLoc(),
                                           builder.getIndexType(), upperBound)
              .getResult();
    }

    // Create scf.for operation
    auto forOp = mlir::scf::ForOp::create(builder, builder.getUnknownLoc(),
                                          lowerBound, upperBound, step);

    builder.setInsertionPointToStart(forOp.getBody());

    // Compute actual induction variable value
    mlir::Value inductionValue = forOp.getInductionVar();
    mlir::Type origType = convertType(builder, inductionVar->getType());

    if (!isIncrementing) {
      mlir::Value one = mlir::arith::ConstantOp::create(
                            builder, builder.getUnknownLoc(),
                            builder.getIndexType(), builder.getIndexAttr(1))
                            .getResult();

      mlir::Value adjustedUpper =
          mlir::arith::SubIOp::create(builder, builder.getUnknownLoc(),
                                      upperBound, one)
              .getResult();

      mlir::Value offset =
          mlir::arith::SubIOp::create(builder, builder.getUnknownLoc(),
                                      inductionValue, lowerBound)
              .getResult();

      inductionValue =
          mlir::arith::SubIOp::create(builder, builder.getUnknownLoc(),
                                      adjustedUpper, offset)
              .getResult();
    }

    // Cast to original type if needed
    if (!origType.isIndex()) {
      inductionValue =
          mlir::arith::IndexCastOp::create(builder, builder.getUnknownLoc(),
                                           origType, inductionValue)
              .getResult();
    }

    // Store induction variable value
    if (symbolTable.count(inductionVar)) {
      mlir::Value memref = symbolTable[inductionVar];
      mlir::memref::StoreOp::create(builder, builder.getUnknownLoc(),
                                    inductionValue, memref, mlir::ValueRange{});
    }

    // Traverse loop body
    loopStack_.push_back({forOp.getBody(), nullptr});
    TraverseStmt(forStmt->getBody());
    loopStack_.pop_back();

    // Ensure terminator exists
    builder.setInsertionPointToEnd(forOp.getBody());
    if (forOp.getBody()->empty() ||
        !forOp.getBody()->back().hasTrait<mlir::OpTrait::IsTerminator>()) {
      mlir::scf::YieldOp::create(builder, builder.getUnknownLoc());
    }

    builder.setInsertionPointAfter(forOp);

  } else {
    auto whileOp =
        mlir::scf::WhileOp::create(builder, builder.getUnknownLoc(),
                                   mlir::TypeRange{}, mlir::ValueRange{});

    // Before region: condition evaluation
    mlir::Block *beforeBlock = &whileOp.getBefore().front();
    builder.setInsertionPointToStart(beforeBlock);

    mlir::Value condition;
    if (forStmt->getCond()) {
      condition = generateExpr(forStmt->getCond());
      condition = convertToBool(condition);
    } else {
      // No condition = infinite loop
      condition = mlir::arith::ConstantOp::create(
                      builder, builder.getUnknownLoc(), builder.getI1Type(),
                      builder.getBoolAttr(true))
                      .getResult();
    }

    mlir::scf::ConditionOp::create(builder, builder.getUnknownLoc(), condition,
                                   mlir::ValueRange{});

    // After region: loop body + increment
    mlir::Block *afterBlock = &whileOp.getAfter().front();
    builder.setInsertionPointToStart(afterBlock);

    loopStack_.push_back({beforeBlock, afterBlock});

    // Execute loop body
    TraverseStmt(forStmt->getBody());

    // Execute increment expression
    if (forStmt->getInc()) {
      generateExpr(forStmt->getInc());
    }

    loopStack_.pop_back();

    // Ensure terminator exists
    builder.setInsertionPointToEnd(afterBlock);
    if (afterBlock->empty() ||
        !afterBlock->back().hasTrait<mlir::OpTrait::IsTerminator>()) {
      mlir::scf::YieldOp::create(builder, builder.getUnknownLoc(),
                                 mlir::ValueRange{});
    }

    builder.setInsertionPointAfter(whileOp);
  }

  return true;
}

} // namespace cmlirc
