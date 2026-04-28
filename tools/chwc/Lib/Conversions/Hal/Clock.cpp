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

  for (auto &[fieldDecl, fieldInfo] : fieldTable) {
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
