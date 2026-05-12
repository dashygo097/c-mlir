#include "../../../Converter.h"
#include "../../Utils/Constants.h"
#include "../../Utils/Numerics.h"
#include "clang/AST/OperationKinds.h"
#include "llvm/Support/WithColor.h"

namespace cmlirc {
auto CMLIRConverter::generateUnaryOperator(clang::UnaryOperator *unOp)
    -> mlir::Value {
  mlir::OpBuilder &builder = contextManager.Builder();
  mlir::Location loc = builder.getUnknownLoc();
  clang::Expr *subExpr = unOp->getSubExpr();

  using CUO = clang::UnaryOperatorKind;

  switch (unOp->getOpcode()) {

  // Identity
  case CUO::UO_Plus:
    return generateExpr(subExpr);

  // Arithmetic negation
  case CUO::UO_Minus: {
    mlir::Value v = generateExpr(subExpr);
    return utils::neg(builder, loc, v);

    return nullptr;
  }

  // Increment / decrement
  case CUO::UO_PreInc:
    return generateIncDecUnaryOperator(subExpr, true, true);
  case CUO::UO_PostInc:
    return generateIncDecUnaryOperator(subExpr, true, false);
  case CUO::UO_PreDec:
    return generateIncDecUnaryOperator(subExpr, false, true);
  case CUO::UO_PostDec:
    return generateIncDecUnaryOperator(subExpr, false, false);

  // Logical NOT: operand == 0
  case CUO::UO_LNot: {
    mlir::Value v = generateExpr(subExpr);
    mlir::Type type = v.getType();
    return utils::emitCmpOp(builder, loc, mlir::arith::CmpIPredicate::eq,
                            mlir::arith::CmpFPredicate::OEQ, v,
                            utils::numericConst(builder, loc, type, 0, 0.0));
  }

  // Bitwise NOT: value ^ ~0
  case CUO::UO_Not: {
    mlir::Value v = generateExpr(subExpr);
    return utils::bitNot(builder, loc, v);
  }

  // Dereference
  case CUO::UO_Deref: {
    mlir::Value base = generateExpr(subExpr);
    auto memrefTy = mlir::dyn_cast<mlir::MemRefType>(base.getType());
    if (!memrefTy) {
      return nullptr;
    }
    if (memrefTy.getRank() == 0) {
      return base; // scalar memref
    }
    lastArrayAccess =
        ArrayAccessInfo{base, {utils::indexConst(builder, loc, 0)}};
    return base;
  }

  // Address-of
  case CUO::UO_AddrOf:
    return generateAddrOfUnaryOperator(subExpr);

  default:
    llvm::WithColor::error()
        << "cmlirc: unsupported unary operator: "
        << clang::UnaryOperator::getOpcodeStr(unOp->getOpcode()) << "\n";
    return nullptr;
  }
}

} // namespace cmlirc
