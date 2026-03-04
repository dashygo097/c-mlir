#include "../../../Converter.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"

namespace cmlirc {

bool CMLIRConverter::TraverseBreakStmt(clang::BreakStmt *) {
  if (!currentFunc || loopStack.empty())
    return true;

  mlir::OpBuilder &builder = context_manager_.Builder();
  mlir::Location loc = builder.getUnknownLoc();

  LoopContext &ctx = loopStack.back();

  mlir::Value trueVal =
      mlir::arith::ConstantOp::create(builder, loc, builder.getI1Type(),
                                      builder.getBoolAttr(true))
          .getResult();
  mlir::memref::StoreOp::create(builder, loc, trueVal, ctx.breakFlag,
                                mlir::ValueRange{});

  mlir::scf::YieldOp::create(builder, loc, mlir::ValueRange{});
  return true;
}

} // namespace cmlirc
