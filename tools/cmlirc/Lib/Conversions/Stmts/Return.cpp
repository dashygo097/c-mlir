#include "../../Converter.h"
#include "../Utils/Constants.h"

namespace cmlirc {

bool CMLIRConverter::TraverseReturnStmt(clang::ReturnStmt *stmt) {
  if (!currentFunc)
    return true;

  mlir::OpBuilder &builder = contextManager.Builder();
  mlir::Location loc = builder.getUnknownLoc();

  mlir::Value retValue;
  if (auto *retExpr = stmt->getRetValue())
    retValue = generateExpr(retExpr);

  if (returnValueCapture) {
    *returnValueCapture = retValue;
    return true;
  }

  for (auto it = loopStack.rbegin(); it != loopStack.rend(); ++it) {
    if (!it->returnFlag)
      continue;

    if (retValue && it->returnValueSlot)
      mlir::memref::StoreOp::create(builder, loc, retValue, it->returnValueSlot,
                                    mlir::ValueRange{});

    mlir::memref::StoreOp::create(builder, loc,
                                  detail::boolConst(builder, loc, true),
                                  it->returnFlag, mlir::ValueRange{});

    return true;
  }

  mlir::func::ReturnOp::create(
      builder, loc, retValue ? mlir::ValueRange{retValue} : mlir::ValueRange{});
  return true;
}

} // namespace cmlirc
