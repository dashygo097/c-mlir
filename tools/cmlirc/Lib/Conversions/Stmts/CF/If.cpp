#include "../../../Converter.h"
#include "../../Utils/Casts.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"

namespace cmlirc {

bool branchEndsWithReturn(clang::Stmt *stmt) {
  if (!stmt)
    return false;
  if (mlir::isa<clang::ReturnStmt>(stmt))
    return true;
  if (auto *compound = mlir::dyn_cast<clang::CompoundStmt>(stmt)) {
    if (compound->body_empty())
      return false;
    return branchEndsWithReturn(compound->body_back());
  }
  if (auto *ifStmt = mlir::dyn_cast<clang::IfStmt>(stmt)) {
    return branchEndsWithReturn(ifStmt->getThen()) &&
           (ifStmt->getElse() ? branchEndsWithReturn(ifStmt->getElse())
                              : false);
  }
  return false;
}

bool CMLIRConverter::TraverseIfStmt(clang::IfStmt *ifStmt) {
  if (!currentFunc)
    return true;

  mlir::OpBuilder &builder = context_manager_.Builder();
  mlir::Location loc = builder.getUnknownLoc();

  mlir::Value condition = generateExpr(ifStmt->getCond());
  if (!condition) {
    llvm::errs() << "Failed to generate if condition\n";
    return false;
  }
  mlir::Value condBool = detail::toBool(builder, loc, condition);

  bool hasElse = ifStmt->getElse() != nullptr;
  bool thenHasReturn = branchEndsWithReturn(ifStmt->getThen());
  bool elseHasReturn = hasElse && branchEndsWithReturn(ifStmt->getElse());

  mlir::Block *currentBlock = builder.getInsertionBlock();
  mlir::Region *region = currentBlock->getParent();

  mlir::Block *mergeBlock = builder.createBlock(region);
  mlir::Block *thenBlock = builder.createBlock(region);
  mlir::Block *elseBlock = hasElse ? builder.createBlock(region) : mergeBlock;

  builder.setInsertionPointToEnd(currentBlock);
  mlir::cf::CondBranchOp::create(builder, loc, condBool, thenBlock,
                                 mlir::ValueRange{}, elseBlock,
                                 mlir::ValueRange{});

  {
    mlir::OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPointToStart(thenBlock);

    TraverseStmt(ifStmt->getThen());

    builder.setInsertionPointToEnd(thenBlock);
    if (thenBlock->empty() ||
        !thenBlock->back().hasTrait<mlir::OpTrait::IsTerminator>())
      mlir::cf::BranchOp::create(builder, loc, mergeBlock, mlir::ValueRange{});
  }

  if (hasElse) {
    mlir::OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPointToStart(elseBlock);

    TraverseStmt(ifStmt->getElse());

    builder.setInsertionPointToEnd(elseBlock);
    if (elseBlock->empty() ||
        !elseBlock->back().hasTrait<mlir::OpTrait::IsTerminator>())
      mlir::cf::BranchOp::create(builder, loc, mergeBlock, mlir::ValueRange{});
  }

  builder.setInsertionPointToStart(mergeBlock);

  (void)thenHasReturn;
  (void)elseHasReturn;

  return true;
}

} // namespace cmlirc
