#include "../../Converter.h"
#include "../Utils/Array.h"

namespace chwc {

void CHWConverter::collectResetValues() {
  mlir::OpBuilder &builder = contextManager.Builder();
  mlir::Location loc = builder.getUnknownLoc();

  for (const clang::FieldDecl *fieldDecl : hardwareFieldOrder) {
    auto fieldIt = fieldTable.find(fieldDecl);
    if (fieldIt == fieldTable.end()) {
      continue;
    }

    HWFieldInfo &fieldInfo = fieldIt->second;
    if (fieldInfo.kind != HWFieldKind::Reg) {
      continue;
    }

    fieldInfo.resetValue = utils::zeroFieldValue(builder, loc, fieldInfo);
  }

  llvm::DenseMap<const clang::VarDecl *, mlir::Value> savedLocalValueTable =
      localValueTable;
  llvm::DenseMap<const clang::VarDecl *, int64_t> savedLocalConstIntTable =
      localConstIntTable;

  bool savedResetMode = isCollectingReset;
  isCollectingReset = true;

  for (clang::CXXMethodDecl *resetMethod : resetMethods) {
    if (!resetMethod || !resetMethod->hasBody()) {
      continue;
    }

    TraverseStmt(resetMethod->getBody());
  }

  isCollectingReset = savedResetMode;
  localValueTable = std::move(savedLocalValueTable);
  localConstIntTable = std::move(savedLocalConstIntTable);

  for (const clang::FieldDecl *fieldDecl : hardwareFieldOrder) {
    auto fieldIt = fieldTable.find(fieldDecl);
    if (fieldIt == fieldTable.end()) {
      continue;
    }

    HWFieldInfo &fieldInfo = fieldIt->second;
    if (fieldInfo.kind != HWFieldKind::Reg) {
      continue;
    }

    if (!fieldInfo.resetValue) {
      fieldInfo.resetValue = utils::zeroFieldValue(builder, loc, fieldInfo);
    }
  }
}

} // namespace chwc
