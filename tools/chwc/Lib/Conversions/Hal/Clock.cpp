#include "../../Converter.h"
#include "../Utils/State.h"

namespace chwc {

void CHWConverter::emitClockTick() {
  if (!clockTickMethod || !clockTickMethod->hasBody()) {
    return;
  }

  TraverseStmt(clockTickMethod->getBody());

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

    mlir::Value nextValue = nextFieldValueTable.lookup(fieldDecl);
    if (!nextValue) {
      nextValue = currentFieldValueTable.lookup(fieldDecl);
    }

    utils::emitRegisterNextAssign(builder, loc, fieldInfo, nextValue);
  }
}

} // namespace chwc
