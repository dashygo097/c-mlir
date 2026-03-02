#include "../../../Converter.h"
#include "../../Utils/Casts.h"
#include "mlir/Dialect/SCF/IR/SCF.h"

namespace cmlirc {

bool branchEndsWithReturn(clang::Stmt *stmt) {
  if (!stmt)
    return false;

  if (mlir::isa<clang::ReturnStmt>(stmt)) {
    return true;
  }

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

// FIXME: Only support either non-return or both return in thenBlock and
// elseBlock
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
  bool bothReturn = thenHasReturn && elseHasReturn;
  bool isNested = (returnValueCapture != nullptr);

  llvm::SmallVector<mlir::Type, 1> resultTypes;
  if (bothReturn && currentFunc.getFunctionType().getNumResults() > 0)
    resultTypes.push_back(currentFunc.getFunctionType().getResult(0));

  auto ifOp = mlir::scf::IfOp::create(
      builder, loc, mlir::TypeRange{resultTypes}, condBool, hasElse);

  mlir::Block *thenBlock = &ifOp.getThenRegion().front();
  builder.setInsertionPointToStart(thenBlock);

  mlir::Value thenReturnValue = nullptr;
  mlir::Value *savedReturnCapture = returnValueCapture;
  if (bothReturn)
    returnValueCapture = &thenReturnValue;

  TraverseStmt(ifStmt->getThen());

  returnValueCapture = savedReturnCapture;

  builder.setInsertionPointToEnd(thenBlock);

  if (thenBlock->empty() ||
      !thenBlock->back().hasTrait<mlir::OpTrait::IsTerminator>()) {
    if (bothReturn && thenReturnValue)
      mlir::scf::YieldOp::create(builder, loc, thenReturnValue);
    else
      mlir::scf::YieldOp::create(builder, builder.getUnknownLoc());
  } else if (bothReturn && mlir::isa<mlir::func::ReturnOp>(thenBlock->back())) {
    auto returnOp = llvm::cast<mlir::func::ReturnOp>(thenBlock->back());
    mlir::ValueRange returnOperands = returnOp.getOperands();
    returnOp.erase();
    if (!returnOperands.empty())
      mlir::scf::YieldOp::create(builder, loc, returnOperands[0]);
    else
      mlir::scf::YieldOp::create(builder, builder.getUnknownLoc());
  }

  if (hasElse) {
    mlir::Block *elseBlock = &ifOp.getElseRegion().front();
    builder.setInsertionPointToStart(elseBlock);

    mlir::Value elseReturnValue = nullptr;
    savedReturnCapture = returnValueCapture;
    if (bothReturn)
      returnValueCapture = &elseReturnValue;

    TraverseStmt(ifStmt->getElse());

    returnValueCapture = savedReturnCapture;

    builder.setInsertionPointToEnd(elseBlock);

    if (elseBlock->empty() ||
        !elseBlock->back().hasTrait<mlir::OpTrait::IsTerminator>()) {
      if (bothReturn && elseReturnValue)
        mlir::scf::YieldOp::create(builder, loc, elseReturnValue);
      else
        mlir::scf::YieldOp::create(builder, builder.getUnknownLoc());
    } else if (bothReturn &&
               mlir::isa<mlir::func::ReturnOp>(elseBlock->back())) {
      auto returnOp = llvm::cast<mlir::func::ReturnOp>(elseBlock->back());
      mlir::ValueRange returnOperands = returnOp.getOperands();
      returnOp.erase();
      if (!returnOperands.empty())
        mlir::scf::YieldOp::create(builder, loc, returnOperands[0]);
      else
        mlir::scf::YieldOp::create(builder, builder.getUnknownLoc());
    }
  }

  builder.setInsertionPointAfter(ifOp);

  if (bothReturn && ifOp.getNumResults() > 0) {
    if (isNested && savedReturnCapture)
      *savedReturnCapture = ifOp.getResult(0);
    else
      mlir::func::ReturnOp::create(builder, loc, ifOp.getResult(0));
  }

  return true;
}

} // namespace cmlirc
