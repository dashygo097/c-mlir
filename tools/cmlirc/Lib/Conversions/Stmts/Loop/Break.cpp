#include "../../../Converter.h"
#include "../../Utils/Constants.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"

namespace cmlirc {

auto CMLIRConverter::TraverseBreakStmt(clang::BreakStmt *) -> bool {
  if (!currentFunc || loopStack.empty()) {
    return true;
  }

  mlir::OpBuilder &builder = contextManager.Builder();
  mlir::Location loc = builder.getUnknownLoc();

  LoopContext &ctx = loopStack.back();

  mlir::Value trueVal = utils::boolConst(builder, loc, true);
  mlir::memref::StoreOp::create(builder, loc, trueVal, ctx.breakFlag,
                                mlir::ValueRange{});

  mlir::scf::YieldOp::create(builder, loc, mlir::ValueRange{});
  return true;
}

} // namespace cmlirc
