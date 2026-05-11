#include "../../Converter.h"
#include "../Utils/Constants.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "llvm/Support/WithColor.h"

namespace cmlirc {

auto CMLIRConverter::TraverseReturnStmt(clang::ReturnStmt *stmt) -> bool {
  if (!currentFunc) {
    return true;
  }

  mlir::OpBuilder &builder = contextManager.Builder();
  mlir::Location loc = builder.getUnknownLoc();

  mlir::Block *block = builder.getInsertionBlock();
  if (block && !block->empty() &&
      block->back().hasTrait<mlir::OpTrait::IsTerminator>()) {
    return true;
  }

  mlir::Value retValue;
  if (auto *retExpr = stmt->getRetValue()) {
    retValue = generateExpr(retExpr);
    if (!retValue) {
      llvm::WithColor::error() << "cmlirc: failed to generate return value\n";
      return false;
    }
  }

  if (returnValueCapture) {
    *returnValueCapture = retValue;
    return true;
  }

  for (auto it = loopStack.rbegin(); it != loopStack.rend(); ++it) {
    if (!it->returnFlag) {
      continue;
    }

    if (retValue && it->returnValueSlot) {
      mlir::memref::StoreOp::create(builder, loc, retValue, it->returnValueSlot,
                                    mlir::ValueRange{});
    }

    mlir::memref::StoreOp::create(builder, loc,
                                  utils::boolConst(builder, loc, true),
                                  it->returnFlag, mlir::ValueRange{});
    return true;
  }

  mlir::func::ReturnOp::create(
      builder, loc, retValue ? mlir::ValueRange{retValue} : mlir::ValueRange{});
  return true;
}

} // namespace cmlirc
