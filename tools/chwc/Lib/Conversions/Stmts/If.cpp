#include "../../Converter.h"
#include "../Utils/Cast.h"
#include "../Utils/Comb.h"

namespace chwc {

auto changedFrom(llvm::DenseMap<const clang::FieldDecl *, mlir::Value> &after,
                 llvm::DenseMap<const clang::FieldDecl *, mlir::Value> &before,
                 const clang::FieldDecl *fieldDecl) -> bool {
  mlir::Value afterValue = after.lookup(fieldDecl);
  mlir::Value beforeValue = before.lookup(fieldDecl);

  if (!afterValue) {
    return false;
  }

  return afterValue != beforeValue;
}

auto CHWConverter::TraverseIfStmt(clang::IfStmt *ifStmt) -> bool {
  mlir::OpBuilder &builder = contextManager.Builder();
  mlir::Location loc = builder.getUnknownLoc();

  mlir::Value cond = generateExpr(ifStmt->getCond());
  if (!cond) {
    return true;
  }

  cond = utils::toBool(builder, loc, cond);
  if (!cond) {
    return true;
  }

  auto savedNextTable = nextFieldValueTable;
  auto savedOutputTable = outputValueTable;

  TraverseStmt(ifStmt->getThen());

  auto thenNextTable = nextFieldValueTable;
  auto thenOutputTable = outputValueTable;

  nextFieldValueTable = savedNextTable;
  outputValueTable = savedOutputTable;

  if (ifStmt->getElse()) {
    TraverseStmt(ifStmt->getElse());
  }

  auto elseNextTable = nextFieldValueTable;
  auto elseOutputTable = outputValueTable;

  nextFieldValueTable = savedNextTable;
  outputValueTable = savedOutputTable;

  for (auto &[fieldDecl, fieldInfo] : fieldTable) {
    if (fieldInfo.kind == HWFieldKind::Reg) {
      bool touched = changedFrom(thenNextTable, savedNextTable, fieldDecl) ||
                     changedFrom(elseNextTable, savedNextTable, fieldDecl);

      if (!touched) {
        continue;
      }

      mlir::Value baseValue = savedNextTable.lookup(fieldDecl);
      if (!baseValue) {
        baseValue = currentFieldValueTable.lookup(fieldDecl);
      }

      mlir::Value thenValue = thenNextTable.lookup(fieldDecl);
      if (!thenValue) {
        thenValue = baseValue;
      }

      mlir::Value elseValue = elseNextTable.lookup(fieldDecl);
      if (!elseValue) {
        elseValue = baseValue;
      }

      nextFieldValueTable[fieldDecl] =
          utils::mux(builder, loc, cond, thenValue, elseValue);
      continue;
    }

    if (fieldInfo.kind == HWFieldKind::Output) {
      bool touched =
          changedFrom(thenOutputTable, savedOutputTable, fieldDecl) ||
          changedFrom(elseOutputTable, savedOutputTable, fieldDecl);

      if (!touched) {
        continue;
      }

      mlir::Value baseValue = savedOutputTable.lookup(fieldDecl);
      if (!baseValue) {
        baseValue = utils::zeroValue(builder, loc, fieldInfo.type);
      }

      mlir::Value thenValue = thenOutputTable.lookup(fieldDecl);
      if (!thenValue) {
        thenValue = baseValue;
      }

      mlir::Value elseValue = elseOutputTable.lookup(fieldDecl);
      if (!elseValue) {
        elseValue = baseValue;
      }

      outputValueTable[fieldDecl] =
          utils::mux(builder, loc, cond, thenValue, elseValue);
      continue;
    }
  }

  return true;
}

} // namespace chwc
