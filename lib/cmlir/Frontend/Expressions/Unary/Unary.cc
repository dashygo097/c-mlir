#include "../../Lowering.h"
#include "clang-mlir/Casts.h"
#include "clang-mlir/Constants.h"
#include "clang-mlir/Numerics.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "clang/AST/OperationKinds.h"
#include "llvm/Support/WithColor.h"

namespace cmlir::frontend {

auto ASTLowering::generateUnaryOperator(clang::UnaryOperator *unOp)
    -> mlir::Value {
  mlir::OpBuilder &builder = context.builder();
  mlir::Location loc = context.source().location(unOp->getExprLoc());
  clang::Expr *subExpr = unOp->getSubExpr();

  using CUO = clang::UnaryOperatorKind;

  switch (unOp->getOpcode()) {
  case CUO::UO_Plus:
    return generateExpr(subExpr);

  case CUO::UO_Minus: {
    mlir::Value value = generateExpr(subExpr);
    return clang_mlir::neg(builder, loc, value);
  }

  case CUO::UO_PreInc:
    return generateIncDecUnaryOperator(subExpr, true, true);

  case CUO::UO_PostInc:
    return generateIncDecUnaryOperator(subExpr, true, false);

  case CUO::UO_PreDec:
    return generateIncDecUnaryOperator(subExpr, false, true);

  case CUO::UO_PostDec:
    return generateIncDecUnaryOperator(subExpr, false, false);

  case CUO::UO_LNot: {
    mlir::Value value = generateExpr(subExpr);
    value = clang_mlir::toBool(builder, loc, value);

    if (!value) {
      return nullptr;
    }

    mlir::Value one = clang_mlir::boolConst(builder, loc, true);
    return clang_mlir::toValue(
        builder, loc,
        mlir::arith::XOrIOp::create(builder, loc, value, one).getResult(),
        types.convertType(unOp->getType()), false);
  }

  case CUO::UO_Not: {
    mlir::Value value = generateExpr(subExpr);
    return clang_mlir::bitNot(builder, loc, value);
  }

  case CUO::UO_Deref: {
    auto address = emitAddress(unOp);
    return mlir::succeeded(address) ? address->materialize(builder, loc)
                                    : mlir::Value{};
  }

  case CUO::UO_AddrOf:
    return generateAddrOfUnaryOperator(subExpr);

  default:
    llvm::WithColor::error()
        << "clang-to-mlir: unsupported unary operator: "
        << clang::UnaryOperator::getOpcodeStr(unOp->getOpcode()) << "\n";
    return nullptr;
  }
}

} // namespace cmlir::frontend
