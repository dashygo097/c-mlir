#include "clang-mlir/MemrefABI.h"

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
  if (!srcType || !srcType.getLayout().isIdentity() ||
      !memrefIsScalarAddressToPointer(srcType, dstType)) {
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
  auto view = mlir::memref::ReinterpretCastOp::create(
                  builder, loc, staticType, value, offset, sizes, strides)
                  .getResult();
  if (staticType == dstType) {
    return view;
  }
  if (!mlir::memref::CastOp::areCastCompatible(staticType, dstType)) {
    return {};
  }
  return mlir::memref::CastOp::create(builder, loc, dstType, view);
}

auto coerceMemRefForCall(mlir::OpBuilder &builder, mlir::Location loc,
                         mlir::Value value, mlir::MemRefType dstType)
    -> mlir::Value {
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
