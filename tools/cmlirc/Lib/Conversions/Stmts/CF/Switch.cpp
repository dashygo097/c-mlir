#include "../../../Converter.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "clang/AST/Stmt.h"

namespace cmlirc {

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

  mlir::Value switchIdx = mlir::arith::IndexCastOp::create(
                              builder, loc, builder.getIndexType(), switchVal)
                              .getResult();

  llvm::SmallVector<SwitchArm> arms;
  collectArms(sw, arms, astCtx);
  if (arms.empty())
    return true;

  clang::Expr *condBase = sw->getCond()->IgnoreImpCasts();
  if (auto *declRef = llvm::dyn_cast<clang::DeclRefExpr>(condBase))
    if (auto *var = llvm::dyn_cast<clang::VarDecl>(declRef->getDecl())) {
      auto it = symbolTable.find(var);
      if (it != symbolTable.end())
        mlir::memref::StoreOp::create(builder, loc, switchVal, it->second,
                                      mlir::ValueRange{});
    }

  llvm::SmallVector<int64_t> caseValues;
  llvm::SmallVector<size_t> caseArmIdx;
  for (size_t i = 0; i < arms.size(); ++i) {
    if (arms[i].isDefault)
      continue;
    for (int64_t v : arms[i].values) {
      caseValues.push_back(v);
      caseArmIdx.push_back(i);
    }
  }

  int64_t defaultArmIdx = -1;
  for (size_t i = 0; i < arms.size(); ++i)
    if (arms[i].isDefault) {
      defaultArmIdx = (int64_t)i;
      break;
    }

  size_t numCases = caseValues.size();

  auto indexSwitch = mlir::scf::IndexSwitchOp::create(
      builder, loc,
      /*resultTypes=*/mlir::TypeRange{}, switchIdx,
      llvm::ArrayRef<int64_t>(caseValues),
      /*numRegions=*/numCases + 1);

  auto emitArmIntoRegion = [&](mlir::Region &reg, size_t armIdx) {
    mlir::Block *blk = new mlir::Block();
    reg.push_back(blk);
    mlir::OpBuilder::InsertionGuard g(builder);
    builder.setInsertionPointToStart(blk);
    for (clang::Stmt *s : arms[armIdx].stmts)
      TraverseStmt(s);
    builder.setInsertionPointToEnd(builder.getInsertionBlock());
    if (builder.getInsertionBlock()->empty() ||
        !builder.getInsertionBlock()
             ->back()
             .hasTrait<mlir::OpTrait::IsTerminator>())
      mlir::scf::YieldOp::create(builder, loc, mlir::ValueRange{});
  };

  for (size_t ci = 0; ci < numCases; ++ci)
    emitArmIntoRegion(indexSwitch.getCaseRegions()[ci], caseArmIdx[ci]);

  if (defaultArmIdx >= 0) {
    emitArmIntoRegion(indexSwitch.getDefaultRegion(), (size_t)defaultArmIdx);
  } else {
    mlir::Block *blk = new mlir::Block();
    indexSwitch.getDefaultRegion().push_back(blk);
    mlir::OpBuilder::InsertionGuard g(builder);
    builder.setInsertionPointToStart(blk);
    mlir::scf::YieldOp::create(builder, loc, mlir::ValueRange{});
  }

  builder.setInsertionPointAfter(indexSwitch);
  return true;
}

} // namespace cmlirc
