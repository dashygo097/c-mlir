#include "../../Converter.h"

namespace cmlirc {

bool CMLIRConverter::TraverseReturnStmt(clang::ReturnStmt *stmt) {
  if (!currentFunc) {
    return true;
  }

  mlir::OpBuilder &builder = context_manager_.Builder();
  mlir::Location loc = builder.getUnknownLoc();

  mlir::Value retValue = nullptr;
  if (auto *retExpr = stmt->getRetValue()) {
    retValue = generateExpr(retExpr);
  }

  if (returnValueCapture) {
    *returnValueCapture = retValue;
    return true;
  }

  if (retValue) {
    mlir::func::ReturnOp::create(builder, loc, mlir::ValueRange{retValue});
  } else {
    mlir::func::ReturnOp::create(builder, builder.getUnknownLoc());
  }

  return true;
}

} // namespace cmlirc
