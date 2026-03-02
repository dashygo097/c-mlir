#include "../../../Converter.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "clang/AST/Stmt.h"

namespace cmlirc {
struct SwitchArm {
  llvm::SmallVector<int64_t, 2> values; // empty → default
  llvm::SmallVector<clang::Stmt *, 8> stmts;
  bool isDefault{false};
  bool hasBreak{false};
};

static void collectArms(clang::SwitchStmt *sw,
                        llvm::SmallVector<SwitchArm> &arms,
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

  // cf.switch requires i32
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

  // Create all blocks up front
  mlir::Block *currentBlock = builder.getInsertionBlock();
  mlir::Region *region = currentBlock->getParent();

  llvm::SmallVector<mlir::Block *> armBlocks;
  for (size_t i = 0; i < arms.size(); ++i)
    armBlocks.push_back(builder.createBlock(region));
  mlir::Block *mergeBlock = builder.createBlock(region);

  // Identify default block
  mlir::Block *defaultBlock = mergeBlock;
  for (size_t i = 0; i < arms.size(); ++i) {
    if (arms[i].isDefault) {
      defaultBlock = armBlocks[i];
      break;
    }
  }

  // Emit cf.switch in the entry block
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

  // Emit arm bodies
  loopStack.push_back({/*headerBlock=*/nullptr, /*exitBlock=*/mergeBlock});

  for (size_t i = 0; i < arms.size(); ++i) {
    mlir::Block *armBlock = armBlocks[i];
    builder.setInsertionPointToStart(armBlock);

    for (clang::Stmt *s : arms[i].stmts)
      TraverseStmt(s);

    mlir::Block *insertionBlock = builder.getInsertionBlock();
    builder.setInsertionPointToEnd(insertionBlock);

    if (!insertionBlock->empty() &&
        insertionBlock->back().hasTrait<mlir::OpTrait::IsTerminator>())
      continue;

    mlir::Block *fallTarget =
        arms[i].hasBreak
            ? mergeBlock
            : (i + 1 < arms.size() ? armBlocks[i + 1] : mergeBlock);
    mlir::cf::BranchOp::create(builder, loc, fallTarget, mlir::ValueRange{});
  }

  loopStack.pop_back();

  builder.setInsertionPointToStart(mergeBlock);
  return true;
}

} // namespace cmlirc
