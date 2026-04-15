#include "../../../Converter.h"
#include "../../Utils/Casts.h"
#include "../../Utils/StmtUtils.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "clang/AST/Stmt.h"
#include "llvm/Support/WithColor.h"

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
    if (mlir::isa<clang::BreakStmt>(s)) {
      if (!arms.empty())
        arms.back().hasBreak = true;
      return;
    }
    if (!arms.empty())
      arms.back().stmts.push_back(s);
  };

  for (clang::Stmt *s : body->body())
    walk(s);

  for (size_t i = 0; i + 1 < arms.size(); ++i) {
    if (!arms[i].hasBreak) {
      bool armTerminates = false;
      for (clang::Stmt *s : arms[i].stmts)
        if (utils::stmtHasReturnRecursively(s)) {
          armTerminates = true;
          break;
        }
      if (armTerminates)
        continue;

      for (clang::Stmt *s : arms[i + 1].stmts)
        arms[i].stmts.push_back(s);
      arms[i].hasBreak = arms[i + 1].hasBreak;
    }
  }
}

bool CMLIRConverter::TraverseSwitchStmt(clang::SwitchStmt *sw) {
  if (!currentFunc)
    return true;

  mlir::OpBuilder &builder = contextManager.Builder();
  mlir::Location loc = builder.getUnknownLoc();
  clang::ASTContext &astCtx = contextManager.ClangContext();

  llvm::SmallVector<SwitchArm> arms;
  collectArms(sw, arms, astCtx);
  if (arms.empty())
    return true;

  mlir::Value switchVal = generateExpr(sw->getCond());
  if (!switchVal) {
    llvm::WithColor::error() << "cmlirc: failed to generate switch condition\n";
    return false;
  }
  mlir::Value switchIdx = utils::toIndex(builder, loc, switchVal);

  clang::Expr *condBase = sw->getCond()->IgnoreImpCasts();
  if (auto *declRef = llvm::dyn_cast<clang::DeclRefExpr>(condBase))
    if (auto *var = llvm::dyn_cast<clang::VarDecl>(declRef->getDecl())) {
      auto it = symbolTable.find(var);
      if (it != symbolTable.end())
        mlir::memref::StoreOp::create(builder, loc, switchVal, it->second,
                                      mlir::ValueRange{});
    }

  SwitchArm *defaultArm = nullptr;
  for (auto &arm : arms)
    if (arm.isDefault) {
      defaultArm = &arm;
      break;
    }

  llvm::SmallVector<int64_t> caseValues;
  llvm::SmallVector<SwitchArm *> caseArmPerValue;
  for (auto &arm : arms) {
    if (arm.isDefault)
      continue;
    for (int64_t v : arm.values) {
      caseValues.push_back(v);
      caseArmPerValue.push_back(&arm);
    }
  }

  size_t numCases = caseValues.size();

  auto switchOp = mlir::scf::IndexSwitchOp::create(
      builder, loc, mlir::TypeRange{}, switchIdx,
      llvm::ArrayRef<int64_t>(caseValues), static_cast<unsigned>(numCases));

  for (size_t i = 0; i < numCases; ++i) {
    mlir::Region &region = switchOp.getCaseRegions()[i];
    auto *blk = new mlir::Block();
    region.push_back(blk);
    auto savedIP = builder.saveInsertionPoint();
    builder.setInsertionPointToStart(blk);
    for (clang::Stmt *s : caseArmPerValue[i]->stmts)
      TraverseStmt(s);
    mlir::Block *cur = builder.getInsertionBlock();
    builder.setInsertionPointToEnd(cur);
    if (cur->empty() || !cur->back().hasTrait<mlir::OpTrait::IsTerminator>())
      mlir::scf::YieldOp::create(builder, loc, mlir::ValueRange{});
    builder.restoreInsertionPoint(savedIP);
  }

  {
    mlir::Region &region = switchOp.getDefaultRegion();
    auto *blk = new mlir::Block();
    region.push_back(blk);
    auto savedIP = builder.saveInsertionPoint();
    builder.setInsertionPointToStart(blk);
    if (defaultArm)
      for (clang::Stmt *s : defaultArm->stmts)
        TraverseStmt(s);
    mlir::Block *cur = builder.getInsertionBlock();
    builder.setInsertionPointToEnd(cur);
    if (cur->empty() || !cur->back().hasTrait<mlir::OpTrait::IsTerminator>())
      mlir::scf::YieldOp::create(builder, loc, mlir::ValueRange{});
    builder.restoreInsertionPoint(savedIP);
  }

  builder.setInsertionPointAfter(switchOp);
  return true;
}

} // namespace cmlirc
