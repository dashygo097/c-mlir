#include "../../../Converter.h"
#include "../../Utils/Casts.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "llvm/Support/WithColor.h"

namespace cmlirc {
auto isInsideStructuredRegion(mlir::OpBuilder &builder,
                              mlir::func::FuncOp funcOp) -> bool {
  mlir::Block *block = builder.getInsertionBlock();
  if (!block) {
    return false;
  }
  mlir::Region *blockRegion = block->getParent();
  if (!blockRegion) {
    return false;
  }
  return blockRegion != &funcOp.getBody();
}

void removeAutoYield(mlir::Block *block) {
  if (!block->empty() && mlir::isa<mlir::scf::YieldOp>(block->back())) {
    block->back().erase();
  }
}

auto CMLIRConverter::TraverseIfStmt(clang::IfStmt *ifStmt) -> bool {
  if (!currentFunc) {
    return true;
  }

  mlir::OpBuilder &builder = contextManager.Builder();
  mlir::Location loc = builder.getUnknownLoc();

  mlir::Value condition = generateExpr(ifStmt->getCond());
  if (!condition) {
    llvm::WithColor::error() << "cmlirc: failed to generate if condition\n";
    return false;
  }
  mlir::Value condBool = detail::toBool(builder, loc, condition);

  bool hasElse = ifStmt->getElse() != nullptr;

  if (isInsideStructuredRegion(builder, currentFunc)) {
    auto ifOp = mlir::scf::IfOp::create(builder, loc, mlir::TypeRange{},
                                        condBool, hasElse);
    {
      mlir::OpBuilder::InsertionGuard guard(builder);
      mlir::Block *thenBlock = &ifOp.getThenRegion().front();
      removeAutoYield(thenBlock);
      builder.setInsertionPointToStart(thenBlock);
      TraverseStmt(ifStmt->getThen());
      builder.setInsertionPointToEnd(thenBlock);
      if (thenBlock->empty() ||
          !thenBlock->back().hasTrait<mlir::OpTrait::IsTerminator>()) {
        mlir::scf::YieldOp::create(builder, loc, mlir::ValueRange{});
      }
    }
    if (hasElse) {
      mlir::OpBuilder::InsertionGuard guard(builder);
      mlir::Block *elseBlock = &ifOp.getElseRegion().front();
      removeAutoYield(elseBlock);
      builder.setInsertionPointToStart(elseBlock);
      TraverseStmt(ifStmt->getElse());
      builder.setInsertionPointToEnd(elseBlock);
      if (elseBlock->empty() ||
          !elseBlock->back().hasTrait<mlir::OpTrait::IsTerminator>()) {
        mlir::scf::YieldOp::create(builder, loc, mlir::ValueRange{});
      }
    }
    builder.setInsertionPointAfter(ifOp);
    return true;
  }

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
        !thenBlock->back().hasTrait<mlir::OpTrait::IsTerminator>()) {
      mlir::cf::BranchOp::create(builder, loc, mergeBlock, mlir::ValueRange{});
    }
  }
  if (hasElse) {
    mlir::OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPointToStart(elseBlock);
    TraverseStmt(ifStmt->getElse());
    builder.setInsertionPointToEnd(elseBlock);
    if (elseBlock->empty() ||
        !elseBlock->back().hasTrait<mlir::OpTrait::IsTerminator>()) {
      mlir::cf::BranchOp::create(builder, loc, mergeBlock, mlir::ValueRange{});
    }
  }

  builder.setInsertionPointToStart(mergeBlock);
  return true;
}

} // namespace cmlirc
