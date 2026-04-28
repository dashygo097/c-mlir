#include "../../Converter.h"
#include "../Utils/HWOps.h"

namespace chwc {

void CHWConverter::emitStateDecls() {
  mlir::OpBuilder &builder = contextManager.Builder();
  mlir::Location loc = builder.getUnknownLoc();

  for (auto &[fieldDecl, fieldInfo] : fieldTable) {
    switch (fieldInfo.kind) {
    case HWFieldKind::Input: {
      currentFieldValueTable[fieldDecl] =
          utils::getInputValue(builder, loc, fieldInfo);
      break;
    }

    case HWFieldKind::Output: {
      break;
    }

    case HWFieldKind::Reg: {
      mlir::Value value =
          utils::emitRegister(builder, loc, fieldInfo, fieldInfo.resetValue);

      currentFieldValueTable[fieldDecl] = value;
      nextFieldValueTable[fieldDecl] = value;
      break;
    }

    case HWFieldKind::Wire: {
      break;
    }
    }
  }
}

} // namespace chwc
