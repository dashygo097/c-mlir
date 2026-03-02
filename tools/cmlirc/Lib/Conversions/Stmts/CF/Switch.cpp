#include "../../../Converter.h"
#include "../../Utils/Casts.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "clang/AST/Stmt.h"

namespace cmlirc {

struct SwitchArm {
  llvm::SmallVector<int64_t, 2> values;
  llvm::SmallVector<clang::Stmt *, 8> stmts;
  bool isDefault{false};
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
    if (!arms.empty())
      arms.back().stmts.push_back(s);
  };

  for (clang::Stmt *s : body->body())
    walk(s);
}

// Detect fallthrough: arm's last non-null stmt is not break/return
bool armFallsThrough(const SwitchArm &arm) {
  for (auto it = arm.stmts.rbegin(); it != arm.stmts.rend(); ++it) {
    if (!*it)
      continue;
    return !llvm::isa<clang::BreakStmt>(*it) &&
           !llvm::isa<clang::ReturnStmt>(*it);
  }
  return true; // empty arm falls through
}

// Detect continue anywhere in stmt tree (doesn't recurse into nested loops)
bool stmtHasContinue(clang::Stmt *s, int depth = 0) {
  if (!s)
    return false;
  if (llvm::isa<clang::ContinueStmt>(s))
    return true;
  if (depth > 0 && llvm::isa<clang::ForStmt, clang::WhileStmt, clang::DoStmt,
                             clang::SwitchStmt>(s))
    return false;
  for (clang::Stmt *c : s->children())
    if (stmtHasContinue(c, depth + 1))
      return true;
  return false;
}

// Emit one region (case or default)
bool emitSwitchRegion(CMLIRConverter &conv, mlir::OpBuilder &builder,
                      mlir::Location loc, mlir::Region &region,
                      llvm::ArrayRef<clang::Stmt *> stmts) {
  // Ensure region has an entry block
  if (region.empty())
    region.push_back(new mlir::Block());

  mlir::OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPointToStart(&region.front());

  for (clang::Stmt *s : stmts)
    conv.TraverseStmt(s);

  // Terminate if needed — break already emitted scf.yield via TraverseBreakStmt
  mlir::Block *exitBlock = builder.getInsertionBlock();
  builder.setInsertionPointToEnd(exitBlock);
  if (exitBlock->empty() ||
      !exitBlock->back().hasTrait<mlir::OpTrait::IsTerminator>())
    mlir::scf::YieldOp::create(builder, loc, mlir::ValueRange{});

  return true;
}

bool CMLIRConverter::TraverseSwitchStmt(clang::SwitchStmt *sw) {
  if (!currentFunc)
    return true;

  mlir::OpBuilder &builder = context_manager_.Builder();
  mlir::Location loc = builder.getUnknownLoc();
  clang::ASTContext &astCtx = context_manager_.ClangContext();

  // Collect arms
  llvm::SmallVector<SwitchArm> arms;
  collectArms(sw, arms, astCtx);
  if (arms.empty())
    return true;

  // Validate: no fallthrough
  for (size_t i = 0; i + 1 < arms.size(); ++i) {
    if (armFallsThrough(arms[i])) {
      llvm::errs() << "error: switch fallthrough is not supported in scf "
                      "dialect (arm "
                   << i << "). Add break.\n";
      return false;
    }
  }

  // Validate: no continue in any case arm
  for (size_t i = 0; i < arms.size(); ++i) {
    for (clang::Stmt *s : arms[i].stmts) {
      if (stmtHasContinue(s)) {
        llvm::errs() << "error: continue inside switch case is not supported "
                        "in scf dialect.\n";
        return false;
      }
    }
  }

  // Condition → index type
  mlir::Value switchVal = generateExpr(sw->getCond());
  if (!switchVal) {
    llvm::errs() << "error: failed to generate switch condition\n";
    return false;
  }
  if (!mlir::isa<mlir::IntegerType>(switchVal.getType())) {
    llvm::errs() << "error: switch condition must be integer\n";
    return false;
  }
  switchVal = detail::toIndex(builder, loc, switchVal);

  // Separate default vs case arms
  SwitchArm *defaultArm = nullptr;
  llvm::SmallVector<SwitchArm *> caseArms;
  for (auto &arm : arms) {
    if (arm.isDefault)
      defaultArm = &arm;
    else
      caseArms.push_back(&arm);
  }

  // Flatten: one entry per (value, arm)
  llvm::SmallVector<int64_t> caseValues;
  llvm::SmallVector<SwitchArm *> caseArmPerValue;
  for (SwitchArm *arm : caseArms)
    for (int64_t v : arm->values) {
      caseValues.push_back(v);
      caseArmPerValue.push_back(arm);
    }

  // Create scf.index_switch
  auto switchOp = mlir::scf::IndexSwitchOp::create(
      builder, loc,
      /*resultTypes=*/mlir::TypeRange{}, switchVal,
      llvm::ArrayRef<int64_t>(caseValues),
      static_cast<unsigned>(caseValues.size()));

  // Emit case regions (one per value)
  breakStack.push_back({BreakTargetKind::ScfYield, nullptr});

  for (size_t i = 0; i < caseValues.size(); ++i) {
    mlir::Region &region = switchOp.getCaseRegions()[i];
    if (!emitSwitchRegion(*this, builder, loc, region,
                          caseArmPerValue[i]->stmts))
      return false;
  }

  // Emit default region
  {
    mlir::Region &region = switchOp.getDefaultRegion();
    llvm::ArrayRef<clang::Stmt *> stmts =
        defaultArm ? llvm::ArrayRef<clang::Stmt *>(defaultArm->stmts)
                   : llvm::ArrayRef<clang::Stmt *>{};
    if (!emitSwitchRegion(*this, builder, loc, region, stmts))
      return false;
  }

  breakStack.pop_back();

  // Continue after switch
  builder.setInsertionPointAfter(switchOp);
  return true;
}

} // namespace cmlirc
