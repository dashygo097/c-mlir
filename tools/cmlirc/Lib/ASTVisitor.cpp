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

  if (forStmt->getInit()) {
    TraverseStmt(forStmt->getInit());
  }

  bool isSimpleCountingLoop = false;
  mlir::Value lowerBound, upperBound, step;
  const clang::VarDecl *inductionVar = nullptr;

  if (auto *init =
          llvm::dyn_cast_or_null<clang::DeclStmt>(forStmt->getInit())) {
    if (init->isSingleDecl()) {
      if (auto *varDecl =
              llvm::dyn_cast<clang::VarDecl>(init->getSingleDecl())) {
        if (varDecl->hasInit()) {
          if (auto *cond = llvm::dyn_cast_or_null<clang::BinaryOperator>(
                  forStmt->getCond())) {
            if (cond->getOpcode() == clang::BO_LT ||
                cond->getOpcode() == clang::BO_LE) {
              if (auto *inc = llvm::dyn_cast_or_null<clang::UnaryOperator>(
                      forStmt->getInc())) {
                if (inc->getOpcode() == clang::UO_PostInc ||
                    inc->getOpcode() == clang::UO_PreInc) {
                  isSimpleCountingLoop = true;
                  inductionVar = varDecl;

                  if (symbolTable.count(varDecl)) {
                    mlir::Value memref = symbolTable[varDecl];
                    lowerBound = mlir::memref::LoadOp::create(
                        builder, builder.getUnknownLoc(), memref);
                  } else {
                    lowerBound = generateExpr(varDecl->getInit());
                  }

                  upperBound = generateExpr(cond->getRHS());
                  step = mlir::arith::ConstantOp::create(
                      builder, builder.getUnknownLoc(), builder.getIndexType(),
                      builder.getIndexAttr(1));
                }
              }
            }
          }
        }
      }
    }
  }

  if (isSimpleCountingLoop && lowerBound && upperBound && step &&
      inductionVar) {
    auto lowerIdx = mlir::arith::IndexCastOp::create(
        builder, builder.getUnknownLoc(), builder.getIndexType(), lowerBound);
    auto upperIdx = mlir::arith::IndexCastOp::create(
        builder, builder.getUnknownLoc(), builder.getIndexType(), upperBound);

    auto forOp = mlir::scf::ForOp::create(builder, builder.getUnknownLoc(),
                                          lowerIdx, upperIdx, step);

    builder.setInsertionPointToStart(forOp.getBody());

    mlir::Value inductionValue = forOp.getInductionVar();

    mlir::Type origType = convertType(builder, inductionVar->getType());
    if (!origType.isIndex()) {
      inductionValue = mlir::arith::IndexCastOp::create(
          builder, builder.getUnknownLoc(), origType, inductionValue);
    }

    if (symbolTable.count(inductionVar)) {
      mlir::Value memref = symbolTable[inductionVar];
      mlir::memref::StoreOp::create(builder, builder.getUnknownLoc(),
                                    inductionValue, memref);
    }

    loopStack_.push_back({forOp.getBody(), nullptr});
    TraverseStmt(forStmt->getBody());
    loopStack_.pop_back();

    mlir::Block *bodyBlock = forOp.getBody();
    mlir::Block::iterator it = bodyBlock->end();
    bool needsYield = true;

    if (!bodyBlock->empty()) {
      --it;
      if (it->hasTrait<mlir::OpTrait::IsTerminator>()) {
        needsYield = false;
      }
    }

    if (needsYield) {
      mlir::scf::YieldOp::create(builder, builder.getUnknownLoc());
    }

    builder.setInsertionPointAfter(forOp);

  } else {
    auto whileOp =
        mlir::scf::WhileOp::create(builder, builder.getUnknownLoc(),
                                   mlir::TypeRange{}, mlir::ValueRange{});

    mlir::Block *beforeBlock = &whileOp.getBefore().front();
    builder.setInsertionPointToStart(beforeBlock);

    mlir::Value condition;
    if (forStmt->getCond()) {
      condition = generateExpr(forStmt->getCond());
      condition = convertToBool(condition);
    } else {
      condition = mlir::arith::ConstantOp::create(
          builder, builder.getUnknownLoc(), builder.getI1Type(),
          builder.getBoolAttr(true));
    }

    mlir::scf::ConditionOp::create(builder, builder.getUnknownLoc(), condition,
                                   mlir::ValueRange{});

    mlir::Block *afterBlock = &whileOp.getAfter().front();
    builder.setInsertionPointToStart(afterBlock);

    loopStack_.push_back({beforeBlock, afterBlock});

    TraverseStmt(forStmt->getBody());

    if (forStmt->getInc()) {
      generateExpr(forStmt->getInc());
    }

    loopStack_.pop_back();

    mlir::Block::iterator it = afterBlock->end();
    bool needsYield = true;

    if (!afterBlock->empty()) {
      --it;
      if (it->hasTrait<mlir::OpTrait::IsTerminator>()) {
        needsYield = false;
      }
    }

    if (needsYield) {
      mlir::scf::YieldOp::create(builder, builder.getUnknownLoc(),
                                 mlir::ValueRange{});
    }

    builder.setInsertionPointAfter(whileOp);
  }

  return true;
}

} // namespace cmlirc
