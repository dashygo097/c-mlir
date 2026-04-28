#include "../../Converter.h"
#include "../Utils/HWOps.h"

namespace chwc {

void CHWConverter::collectResetValues() {
  mlir::OpBuilder &builder = contextManager.Builder();
  mlir::Location loc = builder.getUnknownLoc();

  for (auto &[fieldDecl, fieldInfo] : fieldTable) {
    if (fieldInfo.kind == HWFieldKind::Reg) {
      fieldInfo.resetValue = utils::zeroValue(builder, loc, fieldInfo.type);
    }
  }

  if (!resetMethod || !resetMethod->hasBody()) {
    return;
  }

  auto *body = mlir::dyn_cast<clang::CompoundStmt>(resetMethod->getBody());
  if (!body) {
    llvm::WithColor::error() << "chwc: reset() body must be compound stmt\n";
    return;
  }

  for (clang::Stmt *stmt : body->body()) {
    auto *assignOp = mlir::dyn_cast<clang::BinaryOperator>(stmt);
    if (!assignOp || !assignOp->isAssignmentOp()) {
      llvm::WithColor::error()
          << "chwc: reset() only supports field = constant\n";
      continue;
    }

    const clang::FieldDecl *fieldDecl = getAssignedField(assignOp->getLHS());
    if (!fieldDecl || !fieldTable.count(fieldDecl)) {
      llvm::WithColor::error()
          << "chwc: reset() assignment lhs must be hardware field\n";
      continue;
    }

    mlir::Value value = generateExpr(assignOp->getRHS());
    if (!value) {
      continue;
    }

    fieldTable[fieldDecl].resetValue = value;
  }
}

} // namespace chwc
