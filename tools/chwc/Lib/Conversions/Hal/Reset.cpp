#include "../../Converter.h"
#include "../Utils/Constant.h"
#include "llvm/Support/WithColor.h"

namespace chwc {

void CHWConverter::collectResetValues() {
  mlir::OpBuilder &builder = contextManager.Builder();
  mlir::Location loc = builder.getUnknownLoc();

  for (auto &[fieldDecl, fieldInfo] : fieldTable) {
    if (fieldInfo.kind == HWFieldKind::Reg) {
      fieldInfo.resetValue = utils::zeroValue(builder, loc, fieldInfo.type);
    }
  }

  for (clang::CXXMethodDecl *resetMethod : resetMethods) {
    if (!resetMethod || !resetMethod->hasBody()) {
      continue;
    }

    auto *body = mlir::dyn_cast<clang::CompoundStmt>(resetMethod->getBody());
    if (!body) {
      llvm::WithColor::error()
          << "chwc: reset method body must be compound stmt\n";
      continue;
    }

    for (clang::Stmt *stmt : body->body()) {
      auto *assignOp = mlir::dyn_cast<clang::BinaryOperator>(stmt);
      if (!assignOp || !assignOp->isAssignmentOp()) {
        llvm::WithColor::error()
            << "chwc: reset method only supports field = constant\n";
        continue;
      }

      const clang::FieldDecl *fieldDecl = getAssignedField(assignOp->getLHS());
      if (!fieldDecl || !fieldTable.count(fieldDecl)) {
        llvm::WithColor::error()
            << "chwc: reset assignment lhs must be hardware field\n";
        continue;
      }

      mlir::Value value = generateExpr(assignOp->getRHS());
      if (!value) {
        continue;
      }

      fieldTable[fieldDecl].resetValue = value;
    }
  }
}

} // namespace chwc
