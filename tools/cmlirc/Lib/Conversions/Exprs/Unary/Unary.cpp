#include "../../../Converter.h"
#include "../../Utils/Constants.h"
#include "clang/AST/OperationKinds.h"

namespace cmlirc {
mlir::Value CMLIRConverter::generateUnaryOperator(clang::UnaryOperator *unOp) {
  mlir::OpBuilder &builder = context_manager_.Builder();
  mlir::Location loc = builder.getUnknownLoc();
  clang::Expr *subExpr = unOp->getSubExpr();

  switch (unOp->getOpcode()) {

  // Identity
  case clang::UO_Plus:
    return generateExpr(subExpr);

  // Arithmetic negation
  case clang::UO_Minus: {
    mlir::Value v = generateExpr(subExpr);
    if (!v)
      return nullptr;
    if (mlir::isa<mlir::IntegerType>(v.getType())) {
      return mlir::arith::SubIOp::create(
                 builder, loc, detail::intConst(builder, loc, v.getType(), 0),
                 v)
          .getResult();
    }
    if (mlir::isa<mlir::FloatType>(v.getType()))
      return mlir::arith::NegFOp::create(builder, loc, v).getResult();

    return nullptr;
  }

  // Increment / decrement
  case clang::UO_PreInc:
    return generateIncDecUnaryOperator(subExpr, /*inc=*/true,
                                       /*pre=*/true);
  case clang::UO_PostInc:
    return generateIncDecUnaryOperator(subExpr, /*inc=*/true,
                                       /*pre=*/false);
  case clang::UO_PreDec:
    return generateIncDecUnaryOperator(subExpr, /*inc=*/false,
                                       /*pre=*/true);
  case clang::UO_PostDec:
    return generateIncDecUnaryOperator(subExpr, /*inc=*/false,
                                       /*pre=*/false);

  // Logical NOT: operand == 0
  case clang::UO_LNot: {
    mlir::Value v = generateExpr(subExpr);
    if (!v)
      return nullptr;
    mlir::Type ty = v.getType();
    if (mlir::isa<mlir::IntegerType>(ty))
      return mlir::arith::CmpIOp::create(builder, loc,
                                         mlir::arith::CmpIPredicate::eq, v,
                                         detail::intConst(builder, loc, ty, 0))
          .getResult();
    if (mlir::isa<mlir::FloatType>(ty))
      return mlir::arith::CmpFOp::create(
                 builder, loc, mlir::arith::CmpFPredicate::OEQ, v,
                 detail::floatConst(builder, loc, ty, 0.0))
          .getResult();
    return nullptr;
  }

  // Bitwise NOT: value ^ ~0
  case clang::UO_Not: {
    mlir::Value v = generateExpr(subExpr);
    if (!v)
      return nullptr;
    if (!mlir::isa<mlir::IntegerType>(v.getType()))
      return nullptr;
    return mlir::arith::XOrIOp::create(
               builder, loc, v, detail::intConst(builder, loc, v.getType(), -1))
        .getResult();
  }

  // Dereference
  case clang::UO_Deref: {
    mlir::Value base = generateExpr(subExpr);
    if (!base)
      return nullptr;
    auto memrefTy = mlir::dyn_cast<mlir::MemRefType>(base.getType());
    if (!memrefTy)
      return nullptr;
    if (memrefTy.getRank() == 0)
      return base; // scalar memref
    lastArrayAccess =
        ArrayAccessInfo{base, {detail::indexConst(builder, loc, 0)}};
    return base;
  }

  // Address-of
  case clang::UO_AddrOf:
    return generateAddrOfUnaryOperator(subExpr);

  default:
    llvm::errs() << "cmlirc: unsupported unary operator: "
                 << clang::UnaryOperator::getOpcodeStr(unOp->getOpcode())
                 << "\n";
    return nullptr;
  }
}

} // namespace cmlirc
