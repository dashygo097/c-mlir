#include "../../Converter.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "clang/AST/Expr.h"

namespace cmlirc {

mlir::Value CMLIRConverter::generateUnaryExprOrTypeTraitExpr(
    clang::UnaryExprOrTypeTraitExpr *expr) {

  clang::ASTContext &astCtx = contextManager.ClangContext();
  mlir::OpBuilder &builder = contextManager.Builder();
  mlir::Location loc = builder.getUnknownLoc();

  using CUETT = clang::UnaryExprOrTypeTrait;

  switch (expr->getKind()) {
  case CUETT::UETT_SizeOf: {
    clang::QualType qty = expr->isArgumentType()
                              ? expr->getArgumentType()
                              : expr->getArgumentExpr()->getType();
    qty = qty.getCanonicalType();
    clang::TypeInfo ti = astCtx.getTypeInfo(qty);
    uint64_t sizeBytes = ti.Width / 8;
    mlir::Type resultType = convertType(expr->getType());
    if (!resultType)
      resultType = builder.getI64Type();
    return mlir::arith::ConstantOp::create(
               builder, loc, resultType,
               builder.getIntegerAttr(resultType,
                                      static_cast<int64_t>(sizeBytes)))
        .getResult();
  }

  case CUETT::UETT_AlignOf: {
    clang::QualType qty = expr->isArgumentType()
                              ? expr->getArgumentType()
                              : expr->getArgumentExpr()->getType();
    qty = qty.getCanonicalType();
    clang::TypeInfo ti = astCtx.getTypeInfo(qty);
    uint64_t alignBytes = ti.Align / 8;
    mlir::Type resultType = convertType(expr->getType());
    if (!resultType)
      resultType = builder.getI64Type();
    return mlir::arith::ConstantOp::create(
               builder, loc, resultType,
               builder.getIntegerAttr(resultType,
                                      static_cast<int64_t>(alignBytes)))
        .getResult();
  }

  default:
    llvm::errs() << "unsupported UnaryExprOrTypeTrait kind: " << expr->getKind()
                 << "\n";
    return {};
  }
}

} // namespace cmlirc
