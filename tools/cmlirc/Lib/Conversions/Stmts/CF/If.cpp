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

  mlir::Region *region = block->getParent();
  if (!region) {
    return false;
  }

  return region != &funcOp.getBody();
}

auto blockHasTerminator(mlir::Block *block) -> bool {
  return block && !block->empty() &&
         block->back().hasTrait<mlir::OpTrait::IsTerminator>();
}

void removeAutoYield(mlir::Block *block) {
  if (block && !block->empty() &&
      mlir::isa<mlir::scf::YieldOp>(block->back())) {
    block->back().erase();
  }
}

auto CMLIRConverter::TraverseIfStmt(clang::IfStmt *ifStmt) -> bool {
  if (!currentFunc) {
    return true;
  }

  mlir::OpBuilder &builder = contextManager.Builder();
  mlir::Location loc = builder.getUnknownLoc();

  mlir::Block *entryBlock = builder.getInsertionBlock();
  if (!entryBlock || blockHasTerminator(entryBlock)) {
    return true;
  }

  mlir::Value condition = generateExpr(ifStmt->getCond());
  if (!condition) {
    llvm::WithColor::error() << "cmlirc: failed to generate if condition\n";
    return false;
  }

  mlir::Value condBool = utils::toBool(builder, loc, condition);
  if (!condBool) {
    llvm::WithColor::error() << "cmlirc: failed to cast if condition to bool\n";
    return false;
  }

  bool hasElse = ifStmt->getElse() != nullptr;

  if (isInsideStructuredRegion(builder, currentFunc)) {
    auto ifOp = mlir::scf::IfOp::create(builder, loc, mlir::TypeRange{},
                                        condBool, hasElse);

    auto emitScfArm = [&](mlir::Block *armBlock, clang::Stmt *body) -> bool {
      mlir::OpBuilder::InsertionGuard guard(builder);

      removeAutoYield(armBlock);
      builder.setInsertionPointToStart(armBlock);

      if (body && !TraverseStmt(body)) {
        return false;
      }

      mlir::Block *exitBlock = builder.getInsertionBlock();
      if (!exitBlock) {
        return false;
      }

      if (exitBlock->getParent() != armBlock->getParent()) {
        llvm::WithColor::error()
            << "cmlirc: structured if arm escaped its region\n";
        return false;
      }

      if (!blockHasTerminator(exitBlock)) {
        builder.setInsertionPointToEnd(exitBlock);
        mlir::scf::YieldOp::create(builder, loc, mlir::ValueRange{});
      }

      return true;
    };

    if (!emitScfArm(&ifOp.getThenRegion().front(), ifStmt->getThen())) {
      return false;
    }

    if (hasElse &&
        !emitScfArm(&ifOp.getElseRegion().front(), ifStmt->getElse())) {
      return false;
    }

    builder.setInsertionPointAfter(ifOp);
    return true;
  }

  mlir::Region *region = entryBlock->getParent();

  mlir::Block *thenBlock = builder.createBlock(region);
  mlir::Block *elseBlock = hasElse ? builder.createBlock(region) : nullptr;
  mlir::Block *mergeBlock = builder.createBlock(region);

  if (!hasElse) {
    elseBlock = mergeBlock;
  }

  builder.setInsertionPointToEnd(entryBlock);
  mlir::cf::CondBranchOp::create(builder, loc, condBool, thenBlock,
                                 mlir::ValueRange{}, elseBlock,
                                 mlir::ValueRange{});

  auto emitCfgArm = [&](mlir::Block *armBlock, clang::Stmt *body) -> bool {
    mlir::OpBuilder::InsertionGuard guard(builder);

    builder.setInsertionPointToStart(armBlock);

    if (body && !TraverseStmt(body)) {
      return false;
    }

    mlir::Block *exitBlock = builder.getInsertionBlock();
    if (!exitBlock) {
      return false;
    }

    if (exitBlock->getParent() != mergeBlock->getParent()) {
      llvm::WithColor::error() << "cmlirc: if arm escaped function CFG\n";
      return false;
    }

    if (!blockHasTerminator(exitBlock)) {
      builder.setInsertionPointToEnd(exitBlock);
      mlir::cf::BranchOp::create(builder, loc, mergeBlock, mlir::ValueRange{});
    }

    return true;
  };

  if (!emitCfgArm(thenBlock, ifStmt->getThen())) {
    return false;
  }

  if (hasElse && !emitCfgArm(elseBlock, ifStmt->getElse())) {
    return false;
  }

  builder.setInsertionPointToStart(mergeBlock);
  return true;
}

} // namespace cmlirc
