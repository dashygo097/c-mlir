#include "../../Converter.h"
#include "../Utils/Casts.h"
#include "../Utils/LHS.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "clang/AST/Expr.h"
#include "llvm/Support/WithColor.h"
#include <functional>

namespace cmlirc {

auto CMLIRConverter::generateArraySubscriptExpr(
    clang::ArraySubscriptExpr *arraySub) -> mlir::Value {
  mlir::OpBuilder &builder = contextManager.Builder();
  mlir::Location loc = builder.getUnknownLoc();

  std::function<mlir::Type(clang::QualType)> toLLVMAggregateType =
      [&](clang::QualType clangType) -> mlir::Type {
    clangType = clangType.getCanonicalType();

    if (auto *arrayType =
            mlir::dyn_cast<clang::ConstantArrayType>(clangType.getTypePtr())) {
      mlir::Type elementType = toLLVMAggregateType(arrayType->getElementType());
      if (!elementType) {
        return nullptr;
      }

      return mlir::LLVM::LLVMArrayType::get(
          elementType, arrayType->getSize().getZExtValue());
    }

    return convertType(clangType);
  };

  auto toI64Index = [&](mlir::Value index) -> mlir::Value {
    if (!index) {
      return nullptr;
    }

    if (index.getType().isIndex()) {
      return mlir::arith::IndexCastOp::create(builder, loc,
                                              builder.getI64Type(), index)
          .getResult();
    }

    if (auto intType = mlir::dyn_cast<mlir::IntegerType>(index.getType())) {
      if (intType.getWidth() == 64) {
        return index;
      }

      return utils::toInteger(builder, loc, index, builder.getI64Type(), false);
    }

    mlir::Value indexValue = utils::toIndex(builder, loc, index);
    if (!indexValue) {
      return nullptr;
    }

    return mlir::arith::IndexCastOp::create(builder, loc, builder.getI64Type(),
                                            indexValue)
        .getResult();
  };

  llvm::SmallVector<mlir::Value, 4> indices;
  clang::Expr *currentExpr = arraySub;

  while (auto *arraySubscript = mlir::dyn_cast<clang::ArraySubscriptExpr>(
             currentExpr->IgnoreParenImpCasts())) {
    mlir::Value idx = generateExpr(arraySubscript->getIdx());
    if (!idx) {
      return nullptr;
    }

    indices.insert(indices.begin(), idx);

    clang::Expr *base = arraySubscript->getBase()->IgnoreParenImpCasts();

    if (auto *implCast = mlir::dyn_cast<clang::ImplicitCastExpr>(base)) {
      if (implCast->getCastKind() == clang::CK_ArrayToPointerDecay ||
          implCast->getCastKind() == clang::CK_NoOp) {
        base = implCast->getSubExpr()->IgnoreParenImpCasts();
      }
    }

    currentExpr = base;
  }

  mlir::Value base = generateExpr(currentExpr);
  if (!base) {
    return nullptr;
  }

  if (mlir::isa<mlir::LLVM::LLVMPointerType>(base.getType())) {
    mlir::Type gepElementType = nullptr;
    llvm::SmallVector<mlir::Value, 4> gepIndices;

    if (currentExpr->getType()->isArrayType()) {
      gepElementType = toLLVMAggregateType(currentExpr->getType());
      gepIndices.push_back(
          utils::intConst(builder, loc, builder.getI64Type(), 0));
    } else {
      gepElementType = convertType(arraySub->getType());
    }

    if (!gepElementType) {
      return nullptr;
    }

    for (mlir::Value idx : indices) {
      mlir::Value indexValue = toI64Index(idx);
      if (!indexValue) {
        llvm::WithColor::error()
            << "cmlirc: failed to convert LLVM array subscript index\n";
        return nullptr;
      }

      gepIndices.push_back(indexValue);
    }

    mlir::Value elementPtr = utils::getLLVMOffsetPointer(
        builder, loc, base, gepElementType, gepIndices);

    if (!elementPtr) {
      return nullptr;
    }

    lastArrayAccess = ArrayAccessInfo{elementPtr, {}};
    return elementPtr;
  }

  llvm::SmallVector<mlir::Value, 4> indexValues;
  indexValues.reserve(indices.size());

  for (mlir::Value idx : indices) {
    mlir::Value indexValue = utils::toIndex(builder, loc, idx);
    if (!indexValue) {
      llvm::WithColor::error()
          << "cmlirc: failed to convert array subscript index to index\n";
      return nullptr;
    }

    indexValues.push_back(indexValue);
  }

  lastArrayAccess = ArrayAccessInfo{base, indexValues};
  return base;
}

} // namespace cmlirc
