#include "../Lowering.h"
#include "clang-mlir/Casts.h"
#include "clang-mlir/Constants.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"

namespace cmlir::frontend {

auto ASTLowering::emitArrayAddress(clang::ArraySubscriptExpr *expr)
    -> mlir::FailureOr<clang_mlir::Address> {
  auto &builder = context.builder();
  auto loc = context.source().location(expr->getExprLoc());
  llvm::SmallVector<mlir::Value, 4> indices;
  llvm::SmallVector<bool, 4> signedIndices;
  clang::ArraySubscriptExpr *current = expr;
  clang::Expr *baseExpr = nullptr;
  for (;;) {
    auto index = generateExpr(current->getIdx());
    if (!index) {
      return mlir::failure();
    }
    indices.insert(indices.begin(), index);
    signedIndices.insert(
        signedIndices.begin(),
        current->getIdx()->getType()->isSignedIntegerOrEnumerationType());
    baseExpr = current->getBase();
    auto *inner = llvm::dyn_cast<clang::ArraySubscriptExpr>(
        baseExpr->IgnoreParenImpCasts());
    if (!inner || !inner->getType()->isArrayType()) {
      break;
    }
    current = inner;
  }
  auto base = generateExpr(baseExpr);
  if (!base) {
    return mlir::failure();
  }
  if (mlir::isa<mlir::LLVM::LLVMPointerType>(base.getType())) {
    auto sourceType = baseExpr->IgnoreParenImpCasts()->getType();
    auto storageType = types.convertStorageType(
        sourceType->isPointerType() ? sourceType->getPointeeType()
                                    : sourceType);
    if (!storageType) {
      return mlir::failure();
    }
    llvm::SmallVector<mlir::Value, 4> offsets;
    if (sourceType->isArrayType()) {
      offsets.push_back(
          clang_mlir::intConst(builder, loc, builder.getI64Type(), 0));
    }
    for (size_t i = 0; i < indices.size(); ++i) {
      auto offset = clang_mlir::toInteger(
          builder, loc, indices[i], builder.getI64Type(), signedIndices[i]);
      if (!offset) {
        return mlir::failure();
      }
      offsets.push_back(offset);
    }
    auto pointer = clang_mlir::getLLVMOffsetPointer(builder, loc, base,
                                                    storageType, offsets);
    auto element = types.convertStorageType(expr->getType());
    return clang_mlir::Address::get(pointer, element);
  }
  auto memref = mlir::dyn_cast<mlir::MemRefType>(base.getType());
  if (!memref) {
    return mlir::failure();
  }
  for (size_t i = 0; i < indices.size(); ++i) {
    indices[i] =
        clang_mlir::toIndex(builder, loc, indices[i], signedIndices[i]);
    if (!indices[i]) {
      return mlir::failure();
    }
  }
  return clang_mlir::Address::get(base, memref.getElementType(), indices);
}

auto ASTLowering::generateArraySubscriptExpr(clang::ArraySubscriptExpr *expr)
    -> mlir::Value {
  auto address = emitArrayAddress(expr);
  if (mlir::failed(address)) {
    return {};
  }
  auto &builder = context.builder();
  auto loc = context.source().location(expr->getExprLoc());
  return expr->getType()->isArrayType() ? address->materialize(builder, loc)
                                        : address->load(builder, loc);
}

} // namespace cmlir::frontend
