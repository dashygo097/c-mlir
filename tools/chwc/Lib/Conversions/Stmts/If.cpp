#include "../../Converter.h"
#include "../Utils/Cast.h"
#include "../Utils/Comb.h"

namespace chwc {

auto CHWConverter::TraverseIfStmt(clang::IfStmt *ifStmt) -> bool {
  if (!ifStmt) {
    return true;
  }

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

  llvm::DenseMap<const clang::FieldDecl *, mlir::Value> savedNext =
      moduleContext.nextValues;
  llvm::DenseMap<const clang::FieldDecl *, mlir::Value> savedOutput =
      moduleContext.outputValues;
  llvm::DenseMap<const clang::FieldDecl *, mlir::Value> savedCurrent =
      moduleContext.currentValues;

  TraverseStmt(ifStmt->getThen());

  llvm::DenseMap<const clang::FieldDecl *, mlir::Value> thenNext =
      moduleContext.nextValues;
  llvm::DenseMap<const clang::FieldDecl *, mlir::Value> thenOutput =
      moduleContext.outputValues;
  llvm::DenseMap<const clang::FieldDecl *, mlir::Value> thenCurrent =
      moduleContext.currentValues;

  moduleContext.nextValues = savedNext;
  moduleContext.outputValues = savedOutput;
  moduleContext.currentValues = savedCurrent;

  if (ifStmt->getElse()) {
    TraverseStmt(ifStmt->getElse());
  }

  llvm::DenseMap<const clang::FieldDecl *, mlir::Value> elseNext =
      moduleContext.nextValues;
  llvm::DenseMap<const clang::FieldDecl *, mlir::Value> elseOutput =
      moduleContext.outputValues;
  llvm::DenseMap<const clang::FieldDecl *, mlir::Value> elseCurrent =
      moduleContext.currentValues;

  moduleContext.nextValues = savedNext;
  moduleContext.outputValues = savedOutput;
  moduleContext.currentValues = savedCurrent;

  auto mergeMap =
      [&](llvm::DenseMap<const clang::FieldDecl *, mlir::Value> &dst,
          const llvm::DenseMap<const clang::FieldDecl *, mlir::Value> &thenMap,
          const llvm::DenseMap<const clang::FieldDecl *, mlir::Value> &elseMap,
          const llvm::DenseMap<const clang::FieldDecl *, mlir::Value>
              &baseMap) {
        for (const clang::FieldDecl *fieldDecl : moduleContext.fieldOrder) {
          mlir::Value thenValue = thenMap.lookup(fieldDecl);
          mlir::Value elseValue = elseMap.lookup(fieldDecl);
          mlir::Value baseValue = baseMap.lookup(fieldDecl);

          if (!thenValue && !elseValue) {
            continue;
          }

          if (!thenValue) {
            thenValue = baseValue;
          }

          if (!elseValue) {
            elseValue = baseValue;
          }

          if (!thenValue || !elseValue) {
            continue;
          }

          dst[fieldDecl] = utils::mux(builder, loc, cond, thenValue, elseValue);
        }
      };

  mergeMap(moduleContext.nextValues, thenNext, elseNext, savedNext);
  mergeMap(moduleContext.outputValues, thenOutput, elseOutput, savedOutput);
  mergeMap(moduleContext.currentValues, thenCurrent, elseCurrent, savedCurrent);

  return true;
}

} // namespace chwc
