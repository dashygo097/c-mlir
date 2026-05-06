#include "../../Converter.h"
#include "../Utils/Comb.h"
#include "llvm/Support/WithColor.h"

namespace chwc {

auto CHWConverter::generateConditionalOperator(
    clang::ConditionalOperator *condOp) -> mlir::Value {
  mlir::OpBuilder &builder = contextManager.Builder();
  mlir::Location loc = builder.getUnknownLoc();

  mlir::Value cond = generateExpr(condOp->getCond());
  if (!cond) {
    llvm::WithColor::error()
        << "chwc: no condition found in ConditionalOperator\n";
    return nullptr;
  }

  mlir::Value trueValue = generateExpr(condOp->getTrueExpr());
  mlir::Value falseValue = generateExpr(condOp->getFalseExpr());

  return utils::mux(builder, loc, cond, trueValue, falseValue);
}

} // namespace chwc
