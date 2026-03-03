#include "../../../Converter.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "clang/AST/Stmt.h"

namespace cmlirc {

struct SwitchArm {
  llvm::SmallVector<int64_t, 2> values;
  llvm::SmallVector<clang::Stmt *, 8> stmts;
  bool isDefault{false};
  bool hasBreak{false};
};

void collectArms(clang::SwitchStmt *sw, llvm::SmallVector<SwitchArm> &arms,
                 clang::ASTContext &ctx) {
  auto *body = llvm::dyn_cast_or_null<clang::CompoundStmt>(sw->getBody());
  if (!body)
    return;
  std::function<void(clang::Stmt *)> walk = [&](clang::Stmt *s) {
    if (!s)
      return;
    if (auto *cs = llvm::dyn_cast<clang::CaseStmt>(s)) {
      bool merge =
          !arms.empty() && arms.back().stmts.empty() && !arms.back().isDefault;
      if (!merge)
        arms.push_back({});
      clang::Expr::EvalResult result;
      if (cs->getLHS()->EvaluateAsInt(result, ctx))
        arms.back().values.push_back(result.Val.getInt().getSExtValue());
      walk(cs->getSubStmt());
      return;
    }
    if (auto *ds = llvm::dyn_cast<clang::DefaultStmt>(s)) {
      bool merge = !arms.empty() && arms.back().stmts.empty();
      if (!merge)
        arms.push_back({});
      arms.back().isDefault = true;
      walk(ds->getSubStmt());
      return;
    }
    if (llvm::isa<clang::BreakStmt>(s)) {
      if (!arms.empty())
        arms.back().hasBreak = true;
      return;
    }
    if (!arms.empty())
      arms.back().stmts.push_back(s);
  };
  for (clang::Stmt *s : body->body())
    walk(s);
}

static bool isInsideStructuredRegion(mlir::OpBuilder &builder,
                                     mlir::func::FuncOp funcOp) {
  mlir::Block *block = builder.getInsertionBlock();
  if (!block)
    return false;
  mlir::Region *blockRegion = block->getParent();
  if (!blockRegion)
    return false;
  return blockRegion != &funcOp.getBody();
}

static void emitSwitchAsIfCascade(CMLIRConverter &conv,
                                  mlir::OpBuilder &builder, mlir::Location loc,
                                  mlir::Value switchVal,
                                  llvm::SmallVector<SwitchArm> &arms,
                                  const clang::VarDecl *condVar,
                                  mlir::Value condAlloca) {
  mlir::Type i32 = builder.getI32Type();

  if (condVar && condAlloca)
    mlir::memref::StoreOp::create(builder, loc, switchVal, condAlloca,
                                  mlir::ValueRange{});

  for (size_t i = 0; i < arms.size(); ++i) {
    const SwitchArm &arm = arms[i];

    mlir::Value cond;
    if (arm.isDefault) {
      mlir::Value allNonMatch;
      for (size_t j = 0; j < arms.size(); ++j) {
        if (arms[j].isDefault)
          continue;
        for (int64_t v : arms[j].values) {
          mlir::Value cv =
              mlir::arith::ConstantOp::create(
                  builder, loc, i32,
                  builder.getI32IntegerAttr(static_cast<int32_t>(v)))
                  .getResult();
          mlir::Value neq =
              mlir::arith::CmpIOp::create(
                  builder, loc, mlir::arith::CmpIPredicate::ne, switchVal, cv)
                  .getResult();
          allNonMatch =
              allNonMatch
                  ? mlir::arith::AndIOp::create(builder, loc, allNonMatch, neq)
                        .getResult()
                  : neq;
        }
      }
      if (!allNonMatch)
        allNonMatch =
            mlir::arith::ConstantOp::create(builder, loc, builder.getI1Type(),
                                            builder.getBoolAttr(true))
                .getResult();
      cond = allNonMatch;
    } else {
      mlir::Value anyMatch;
      for (int64_t v : arm.values) {
        mlir::Value cv = mlir::arith::ConstantOp::create(
                             builder, loc, i32,
                             builder.getI32IntegerAttr(static_cast<int32_t>(v)))
                             .getResult();
        mlir::Value eq =
            mlir::arith::CmpIOp::create(
                builder, loc, mlir::arith::CmpIPredicate::eq, switchVal, cv)
                .getResult();
        anyMatch = anyMatch
                       ? mlir::arith::OrIOp::create(builder, loc, anyMatch, eq)
                             .getResult()
                       : eq;
      }
      if (!anyMatch)
        continue;
      cond = anyMatch;
    }

    auto ifOp =
        mlir::scf::IfOp::create(builder, loc, mlir::TypeRange{}, cond, false);
    {
      mlir::OpBuilder::InsertionGuard g(builder);
      mlir::Block *thenBlock = &ifOp.getThenRegion().front();
      if (!thenBlock->empty() &&
          mlir::isa<mlir::scf::YieldOp>(thenBlock->back()))
        thenBlock->back().erase();
      builder.setInsertionPointToStart(thenBlock);
      for (clang::Stmt *s : arm.stmts)
        conv.TraverseStmt(s);
      builder.setInsertionPointToEnd(thenBlock);
      if (thenBlock->empty() ||
          !thenBlock->back().hasTrait<mlir::OpTrait::IsTerminator>())
        mlir::scf::YieldOp::create(builder, loc, mlir::ValueRange{});
    }
    builder.setInsertionPointAfter(ifOp);
  }
}

bool CMLIRConverter::TraverseSwitchStmt(clang::SwitchStmt *sw) {
  if (!currentFunc)
    return true;

  mlir::OpBuilder &builder = context_manager_.Builder();
  mlir::Location loc = builder.getUnknownLoc();
  clang::ASTContext &astCtx = context_manager_.ClangContext();

  mlir::Value switchVal = generateExpr(sw->getCond());
  if (!switchVal) {
    llvm::errs() << "switch: bad condition\n";
    return false;
  }

  mlir::Type i32 = builder.getI32Type();
  if (switchVal.getType() != i32) {
    uint32_t w = mlir::cast<mlir::IntegerType>(switchVal.getType()).getWidth();
    if (w < 32)
      switchVal = mlir::arith::ExtSIOp::create(builder, loc, i32, switchVal)
                      .getResult();
    else
      switchVal = mlir::arith::TruncIOp::create(builder, loc, i32, switchVal)
                      .getResult();
  }

  llvm::SmallVector<SwitchArm> arms;
  collectArms(sw, arms, astCtx);
  if (arms.empty())
    return true;

  if (isInsideStructuredRegion(builder, currentFunc)) {
    const clang::VarDecl *condVar = nullptr;
    mlir::Value condAlloca;
    clang::Expr *condBase = sw->getCond()->IgnoreImpCasts();
    if (auto *declRef = llvm::dyn_cast<clang::DeclRefExpr>(condBase)) {
      condVar = llvm::dyn_cast<clang::VarDecl>(declRef->getDecl());
      if (condVar) {
        auto it = symbolTable.find(condVar);
        if (it != symbolTable.end())
          condAlloca = it->second;
      }
    }

    switchStack.push_back({nullptr, nullptr});
    emitSwitchAsIfCascade(*this, builder, loc, switchVal, arms, condVar,
                          condAlloca);
    switchStack.pop_back();
    return true;
  }

  mlir::Block *currentBlock = builder.getInsertionBlock();
  mlir::Region *region = currentBlock->getParent();

  mlir::Block *mergeBlock = builder.createBlock(region);
  llvm::SmallVector<mlir::Block *> armBlocks;
  for (size_t i = 0; i < arms.size(); ++i)
    armBlocks.push_back(builder.createBlock(region));

  mlir::Block *defaultBlock = mergeBlock;
  for (size_t i = 0; i < arms.size(); ++i) {
    if (arms[i].isDefault) {
      defaultBlock = armBlocks[i];
      break;
    }
  }

  llvm::SmallVector<int32_t> caseValues;
  llvm::SmallVector<mlir::Block *> caseBlocks;
  for (size_t i = 0; i < arms.size(); ++i) {
    if (arms[i].isDefault)
      continue;
    for (int64_t v : arms[i].values) {
      caseValues.push_back(static_cast<int32_t>(v));
      caseBlocks.push_back(armBlocks[i]);
    }
  }

  builder.setInsertionPointToEnd(currentBlock);
  mlir::cf::SwitchOp::create(builder, loc, switchVal, defaultBlock,
                             mlir::ValueRange{},
                             llvm::ArrayRef<int32_t>(caseValues),
                             llvm::ArrayRef<mlir::Block *>(caseBlocks),
                             llvm::SmallVector<mlir::ValueRange>(
                                 caseBlocks.size(), mlir::ValueRange{}));

  switchStack.push_back({nullptr, mergeBlock});

  for (size_t i = 0; i < arms.size(); ++i) {
    mlir::Block *armBlock = armBlocks[i];
    builder.setInsertionPointToStart(armBlock);
    for (clang::Stmt *s : arms[i].stmts)
      TraverseStmt(s);
    builder.setInsertionPointToEnd(armBlock);
    if (!armBlock->empty() &&
        armBlock->back().hasTrait<mlir::OpTrait::IsTerminator>())
      continue;
    if (arms[i].hasBreak)
      mlir::cf::BranchOp::create(builder, loc, mergeBlock, mlir::ValueRange{});
    else {
      mlir::Block *fallTarget =
          (i + 1 < arms.size()) ? armBlocks[i + 1] : mergeBlock;
      mlir::cf::BranchOp::create(builder, loc, fallTarget, mlir::ValueRange{});
    }
  }

  switchStack.pop_back();
  builder.setInsertionPointToStart(mergeBlock);
  return true;
}

} // namespace cmlirc
