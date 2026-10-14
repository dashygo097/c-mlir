#include "PointerArithmetic.h"
#include "clang-mlir/Address.h"
#include "clang-mlir/Casts.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Diagnostics.h"

namespace cmlir::frontend {

auto emitPointerOffset(mlir::OpBuilder &builder, mlir::Location loc,
                       mlir::Value pointer, mlir::Value offset,
                       mlir::Type storageElement, bool signedOffset,
                       bool subtract) -> mlir::Value {
  if (!pointer || !offset ||
      (!offset.getType().isIndex() &&
       !mlir::isa<mlir::IntegerType>(offset.getType()))) {
    mlir::emitError(loc, "pointer offset must be an integer");
    return {};
  }
  if (mlir::isa<mlir::LLVM::LLVMPointerType>(pointer.getType())) {
    if (!storageElement || !mlir::LLVM::isCompatibleType(storageElement)) {
      mlir::emitError(loc, "unsupported pointer arithmetic element type");
      return {};
    }
    auto index = clang_mlir::toInteger(builder, loc, offset,
                                       builder.getI64Type(), signedOffset);
    if (!index) {
      return {};
    }
    if (subtract) {
      auto zero = mlir::arith::ConstantIntOp::create(builder, loc, 0, 64);
      index = mlir::arith::SubIOp::create(builder, loc, zero, index);
    }
    return clang_mlir::getLLVMOffsetPointer(builder, loc, pointer,
                                            storageElement, {index});
  }

  auto type = mlir::dyn_cast<mlir::MemRefType>(pointer.getType());
  if (!type || type.getRank() == 0) {
    mlir::emitError(loc, "unsupported pointer arithmetic representation");
    return {};
  }
  llvm::SmallVector<int64_t, 4> staticStrides;
  int64_t staticOffset;
  if (mlir::failed(type.getStridesAndOffset(staticStrides, staticOffset))) {
    mlir::emitError(loc, "pointer arithmetic requires a strided memref");
    return {};
  }
  auto index = clang_mlir::toIndex(builder, loc, offset, signedOffset);
  if (!index) {
    return {};
  }
  if (subtract) {
    auto zero = mlir::arith::ConstantIndexOp::create(builder, loc, 0);
    index = mlir::arith::SubIOp::create(builder, loc, zero, index);
  }
  auto metadata =
      mlir::memref::ExtractStridedMetadataOp::create(builder, loc, pointer);
  auto displacement = mlir::arith::MulIOp::create(
      builder, loc, index, metadata.getStrides().front());
  auto newOffset = mlir::arith::AddIOp::create(
      builder, loc, metadata.getOffset(), displacement);
  auto remaining = mlir::arith::SubIOp::create(
      builder, loc, metadata.getSizes().front(), index);
  llvm::SmallVector<mlir::OpFoldResult, 4> sizes{remaining.getResult()},
      strides;
  for (int64_t dim = 1; dim < type.getRank(); ++dim) {
    sizes.push_back(type.isDynamicDim(dim)
                        ? mlir::OpFoldResult(metadata.getSizes()[dim])
                        : builder.getIndexAttr(type.getDimSize(dim)));
  }
  for (auto [dim, stride] : llvm::enumerate(staticStrides)) {
    strides.push_back(mlir::ShapedType::isDynamic(stride)
                          ? mlir::OpFoldResult(metadata.getStrides()[dim])
                          : builder.getIndexAttr(stride));
  }
  llvm::SmallVector<int64_t, 4> shape(type.getShape());
  shape.front() = mlir::ShapedType::kDynamic;
  auto layout = mlir::StridedLayoutAttr::get(
      builder.getContext(), mlir::ShapedType::kDynamic, staticStrides);
  auto resultType = mlir::MemRefType::get(shape, type.getElementType(), layout,
                                          type.getMemorySpace());
  // Reinterpretation offsets are relative to the underlying buffer. Include
  // the existing descriptor offset so nested p + i - j remains correct.
  return mlir::memref::ReinterpretCastOp::create(
      builder, loc, resultType, metadata.getBaseBuffer(),
      mlir::OpFoldResult(newOffset.getResult()), sizes, strides);
}

} // namespace cmlir::frontend
