#include "../Converter.h"

namespace cmlirc {
bool CMLIRConverter::hasSideEffects(clang::Expr *expr) const {
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

bool CMLIRConverter::branchEndsWithReturn(clang::Stmt *stmt) {
  if (!stmt)
    return false;

  if (llvm::isa<clang::ReturnStmt>(stmt)) {
    return true;
  }

  if (auto *compound = llvm::dyn_cast<clang::CompoundStmt>(stmt)) {
    if (compound->body_empty())
      return false;
    return branchEndsWithReturn(compound->body_back());
  }

  if (auto *ifStmt = llvm::dyn_cast<clang::IfStmt>(stmt)) {
    return branchEndsWithReturn(ifStmt->getThen()) &&
           (ifStmt->getElse() ? branchEndsWithReturn(ifStmt->getElse())
                              : false);
  }

  return false;
}

void CMLIRConverter::storeInitListValues(clang::InitListExpr *initList,
                                         mlir::Value memref) {
  mlir::OpBuilder &builder = context_manager_.Builder();
  mlir::Location loc = builder.getUnknownLoc();

  std::function<void(clang::InitListExpr *,
                     llvm::SmallVector<mlir::Value, 4> &)>
      storeValues = [&](clang::InitListExpr *list,
                        llvm::SmallVector<mlir::Value, 4> &currentIndices) {
        for (unsigned i = 0; i < list->getNumInits(); ++i) {
          clang::Expr *init = list->getInit(i);

          mlir::Value indexVal =
              mlir::arith::ConstantOp::create(
                  builder, loc, builder.getIndexType(), builder.getIndexAttr(i))
                  .getResult();

          if (auto *nestedList = llvm::dyn_cast<clang::InitListExpr>(init)) {
            currentIndices.push_back(indexVal);
            storeValues(nestedList, currentIndices);
            currentIndices.pop_back();
          } else {
            mlir::Value value = generateExpr(init);
            if (!value) {
              llvm::errs() << "Failed to generate init value\n";
              continue;
            }

            llvm::SmallVector<mlir::Value, 4> fullIndices;
            fullIndices.append(currentIndices.begin(), currentIndices.end());
            fullIndices.push_back(indexVal);

            mlir::memref::StoreOp::create(builder, loc, value, memref,
                                          fullIndices);
          }
        }
      };

  llvm::SmallVector<mlir::Value, 4> indices;
  storeValues(initList, indices);
}

} // namespace cmlirc
