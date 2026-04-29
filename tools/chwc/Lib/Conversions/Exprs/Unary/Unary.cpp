#include "../../../Converter.h"
#include "../../Utils/Casts.h"
#include "../../Utils/Comb.h"
#include "clang/AST/OperationKinds.h"
#include "llvm/Support/WithColor.h"

namespace chwc {

auto CHWConverter::generateUnaryOperator(clang::UnaryOperator *unOp)
    -> mlir::Value {
  mlir::OpBuilder &builder = contextManager.Builder();
  mlir::Location loc = builder.getUnknownLoc();

  clang::Expr *subExpr = unOp->getSubExpr();

  using CUO = clang::UnaryOperatorKind;

  switch (unOp->getOpcode()) {
  case CUO::UO_Plus:
    return generateExpr(subExpr);

  case CUO::UO_Minus: {
    mlir::Value value = generateExpr(subExpr);
    if (!value) {
      return nullptr;
    }

    return utils::neg(builder, loc, value);
  }

  case CUO::UO_Not: {
    mlir::Value value = generateExpr(subExpr);
    if (!value) {
      return nullptr;
    }

    return utils::bitNot(builder, loc, value);
  }

  case CUO::UO_LNot: {
    mlir::Value value = generateExpr(subExpr);
    if (!value) {
      return nullptr;
    }

    value = utils::toBool(builder, loc, value);
    if (!value) {
      return nullptr;
    }

    mlir::Value falseValue =
        utils::zeroValue(builder, loc, builder.getI1Type());
    if (!falseValue) {
      return nullptr;
    }

    return utils::icmpEq(builder, loc, value, falseValue);
  }

  case CUO::UO_PreInc:
    return generateIncDecUnaryOperator(subExpr, true, true);

  case CUO::UO_PostInc:
    return generateIncDecUnaryOperator(subExpr, true, false);

  case CUO::UO_PreDec:
    return generateIncDecUnaryOperator(subExpr, false, true);

  case CUO::UO_PostDec:
    return generateIncDecUnaryOperator(subExpr, false, false);

  case CUO::UO_AddrOf:
    return generateAddrOfUnaryOperator(subExpr);

  case CUO::UO_Deref:
    llvm::WithColor::error()
        << "chwc: pointer dereference is unsupported in hardware DSL\n";
    return nullptr;

  default:
    llvm::WithColor::error()
        << "chwc: unsupported unary operator: "
        << clang::UnaryOperator::getOpcodeStr(unOp->getOpcode()) << "\n";
    return nullptr;
  }
}

} // namespace chwc
