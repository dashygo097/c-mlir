#include "clang-mlir/MemrefABI.h"
#include "clang-mlir/Casts.h"
#include "mlir/Conversion/LLVMCommon/MemRefBuilder.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"

namespace clang_mlir {

auto memrefSameElementType(mlir::MemRefType src, mlir::MemRefType dst) -> bool {
  return src.getElementType() == dst.getElementType();
}

auto memrefIsScalarAddressToPointer(mlir::MemRefType src, mlir::MemRefType dst)
    -> bool {
  return src.getRank() == 0 && dst.getRank() == 1 &&
         memrefSameElementType(src, dst);
}

auto reinterpretScalarMemRefAsPointerMemRef(mlir::OpBuilder &builder,
                                            mlir::Location loc,
                                            mlir::Value value,
                                            mlir::MemRefType dstType)
    -> mlir::Value {
  auto srcType = mlir::dyn_cast<mlir::MemRefType>(value.getType());
  if (!srcType || !memrefIsScalarAddressToPointer(srcType, dstType) ||
      srcType.getMemorySpace() != dstType.getMemorySpace() ||
      !dstType.getLayout().isIdentity()) {
    return nullptr;
  }

  mlir::OpFoldResult offset = builder.getIndexAttr(0);

  llvm::SmallVector<mlir::OpFoldResult, 1> sizes;
  llvm::SmallVector<mlir::OpFoldResult, 1> strides;

  sizes.push_back(builder.getIndexAttr(1));
  strides.push_back(builder.getIndexAttr(1));

  auto staticType = mlir::MemRefType::get({1}, dstType.getElementType(),
                                          mlir::MemRefLayoutAttrInterface{},
                                          dstType.getMemorySpace());
  mlir::Value view;
  if (!srcType.getLayout().isIdentity()) {
    // Rebase at the addressed element. A reinterpret_cast offset is relative
    // to the underlying buffer, so resetting it to zero would lose &array[i].
    // Preserve the remaining innermost array extent as well as the pointer.
    // A scalar subview's shape alone would incorrectly limit the callee to one
    // element. The borrowed descriptor conveys no ownership.
    auto subview = value.getDefiningOp<mlir::memref::SubViewOp>();
    if (!subview || !dstType.isDynamicDim(0)) {
      return {};
    }
    auto sourceType =
        mlir::cast<mlir::MemRefType>(subview.getSource().getType());
    auto [sourceStrides, sourceOffset] = sourceType.getStridesAndOffset();
    if (sourceStrides.empty() || sourceStrides.back() != 1) {
      return {};
    }
    mlir::LLVMTypeConverter converter(builder.getContext());
    auto addressSpace = converter.getMemRefAddressSpace(srcType);
    if (mlir::failed(addressSpace)) {
      return {};
    }
    auto pointerType =
        mlir::LLVM::LLVMPointerType::get(builder.getContext(), *addressSpace);
    auto pointer = memrefToPointer(builder, loc, value, pointerType);
    if (!pointer) {
      return {};
    }
    auto descriptor = mlir::MemRefDescriptor::fromStaticShape(
        builder, loc, converter, staticType, pointer);
    auto source = subview.getSource();
    auto dimension =
        mlir::cast<mlir::MemRefType>(source.getType()).getRank() - 1;
    auto extent = mlir::memref::DimOp::create(builder, loc, source, dimension);
    auto mixedOffset = subview.getMixedOffsets().back();
    auto index = llvm::dyn_cast<mlir::Value>(mixedOffset);
    if (!index) {
      index = indexConst(builder, loc,
                         llvm::cast<mlir::IntegerAttr>(
                             llvm::cast<mlir::Attribute>(mixedOffset))
                             .getInt());
    }
    auto remaining = mlir::arith::SubIOp::create(builder, loc, extent, index);
    auto size = mlir::arith::IndexCastOp::create(
        builder, loc, builder.getI64Type(), remaining);
    descriptor.setSize(builder, loc, 0, size);
    return mlir::UnrealizedConversionCastOp::create(
               builder, loc, mlir::TypeRange{dstType},
               mlir::ValueRange{descriptor})
        .getResult(0);
  } else {
    view = mlir::memref::ReinterpretCastOp::create(
               builder, loc, staticType, value, offset, sizes, strides)
               .getResult();
  }
  if (staticType == dstType) {
    return view;
  }
  if (!mlir::memref::CastOp::areCastCompatible(staticType, dstType)) {
    return {};
  }
  return mlir::memref::CastOp::create(builder, loc, dstType, view);
}

auto coerceMemRef(mlir::OpBuilder &builder, mlir::Location loc,
                  mlir::Value value, mlir::MemRefType dstType) -> mlir::Value {
  auto srcType = mlir::dyn_cast<mlir::MemRefType>(value.getType());
  if (!srcType) {
    return nullptr;
  }

  if (srcType == dstType) {
    return value;
  }

  if (!memrefSameElementType(srcType, dstType)) {
    return nullptr;
  }

  if (mlir::memref::CastOp::areCastCompatible(srcType, dstType)) {
    return mlir::memref::CastOp::create(builder, loc, dstType, value)
        .getResult();
  }

  if (mlir::Value reinterpreted = reinterpretScalarMemRefAsPointerMemRef(
          builder, loc, value, dstType)) {
    return reinterpreted;
  }

  return nullptr;
}

} // namespace clang_mlir
