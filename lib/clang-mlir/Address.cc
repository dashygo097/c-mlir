#include "clang-mlir/Address.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Diagnostics.h"

namespace clang_mlir {

auto Address::get(mlir::Value base, mlir::Type element,
                  llvm::ArrayRef<mlir::Value> indices)
    -> mlir::FailureOr<Address> {
  if (!base || !element) {
    return mlir::failure();
  }
  for (auto index : indices) {
    if (!index) {
      return mlir::failure();
    }
  }
  if (auto memref = mlir::dyn_cast<mlir::MemRefType>(base.getType())) {
    if (memref.getRank() != static_cast<int64_t>(indices.size()) ||
        memref.getElementType() != element) {
      return mlir::failure();
    }
    for (auto index : indices) {
      if (!index.getType().isIndex()) {
        return mlir::failure();
      }
    }
  } else if (!mlir::isa<mlir::LLVM::LLVMPointerType>(base.getType())) {
    return mlir::failure();
  } else {
    if (!mlir::LLVM::isCompatibleType(element)) {
      return mlir::failure();
    }
    for (auto index : indices) {
      if (!index.getType().isIndex() &&
          !mlir::isa<mlir::IntegerType>(index.getType())) {
        return mlir::failure();
      }
    }
  }
  return Address(base, element, indices);
}

auto getLLVMOffsetPointer(mlir::OpBuilder &builder, mlir::Location loc,
                          mlir::Value base, mlir::Type element,
                          llvm::ArrayRef<mlir::Value> indices) -> mlir::Value {
  if (!base || !element ||
      !mlir::isa<mlir::LLVM::LLVMPointerType>(base.getType())) {
    return {};
  }
  if (indices.empty()) {
    return base;
  }
  llvm::SmallVector<mlir::LLVM::GEPArg> converted;
  for (auto index : indices) {
    if (!index) {
      return {};
    }
    if (index.getType().isIndex()) {
      index = mlir::arith::IndexCastOp::create(builder, loc,
                                               builder.getI64Type(), index);
    }
    if (!mlir::isa<mlir::IntegerType>(index.getType())) {
      return {};
    }
    converted.push_back(index);
  }
  return mlir::LLVM::GEPOp::create(builder, loc, base.getType(), element, base,
                                   converted);
}

auto Address::materialize(mlir::OpBuilder &builder, mlir::Location loc) const
    -> mlir::Value {
  if (mlir::isa<mlir::LLVM::LLVMPointerType>(base.getType())) {
    return getLLVMOffsetPointer(builder, loc, base, element, indices);
  }
  if (indices.empty()) {
    return base;
  }
  auto type = mlir::cast<mlir::MemRefType>(base.getType());
  llvm::SmallVector<mlir::OpFoldResult> offsets, sizes, strides;
  for (auto index : indices) {
    offsets.push_back(index);
    sizes.push_back(builder.getIndexAttr(1));
    strides.push_back(builder.getIndexAttr(1));
  }
  auto result = mlir::memref::SubViewOp::inferRankReducedResultType(
      llvm::ArrayRef<int64_t>{}, type, offsets, sizes, strides);
  return mlir::memref::SubViewOp::create(builder, loc, result, base, offsets,
                                         sizes, strides);
}

auto Address::load(mlir::OpBuilder &builder, mlir::Location loc) const
    -> mlir::Value {
  if (mlir::isa<mlir::MemRefType>(base.getType())) {
    return mlir::memref::LoadOp::create(builder, loc, base, indices);
  }
  return mlir::LLVM::LoadOp::create(builder, loc, element,
                                    materialize(builder, loc));
}

auto Address::store(mlir::OpBuilder &builder, mlir::Location loc,
                    mlir::Value value) const -> mlir::LogicalResult {
  if (!value || value.getType() != element) {
    mlir::emitError(loc, "stored value does not match address element type");
    return mlir::failure();
  }
  if (mlir::isa<mlir::MemRefType>(base.getType())) {
    mlir::memref::StoreOp::create(builder, loc, value, base, indices);
  } else {
    auto pointer = materialize(builder, loc);
    if (auto record = llvm::dyn_cast<mlir::LLVM::LLVMStructType>(element)) {
      // Field stores expose aggregate initialization to upstream SROA. Whole
      // aggregate stores currently block LLVM's destructuring interfaces.
      for (auto [index, type] : llvm::enumerate(record.getBody())) {
        auto fieldPointer = mlir::LLVM::GEPOp::create(
            builder, loc, pointer.getType(), record, pointer,
            llvm::ArrayRef<mlir::LLVM::GEPArg>{0, static_cast<int32_t>(index)});
        auto fieldValue =
            mlir::LLVM::ExtractValueOp::create(builder, loc, value, index);
        auto field = Address::get(fieldPointer, type);
        if (mlir::failed(field) ||
            mlir::failed(field->store(builder, loc, fieldValue))) {
          return mlir::failure();
        }
      }
    } else {
      mlir::LLVM::StoreOp::create(builder, loc, value, pointer);
    }
  }
  return mlir::success();
}

auto createLocalStorage(mlir::OpBuilder &builder, mlir::Location loc,
                        mlir::Type type, mlir::ValueRange dynamicSizes)
    -> mlir::Value {
  if (!type) {
    return {};
  }
  if (auto memref = mlir::dyn_cast<mlir::MemRefType>(type)) {
    if (memref.getNumDynamicDims() != dynamicSizes.size()) {
      return {};
    }
    return mlir::memref::AllocaOp::create(builder, loc, memref, dynamicSizes);
  }
  if (!dynamicSizes.empty()) {
    return {};
  }
  if (mlir::MemRefType::isValidElementType(type)) {
    return mlir::memref::AllocaOp::create(builder, loc,
                                          mlir::MemRefType::get({}, type));
  }
  if (mlir::LLVM::isCompatibleType(type)) {
    // LLVM's memory-slot interfaces recognize LLVM constants when proving
    // this is a single object, enabling upstream SROA and mem2reg.
    auto one = mlir::LLVM::ConstantOp::create(
        builder, loc, builder.getI64Type(), builder.getI64IntegerAttr(1));
    return mlir::LLVM::AllocaOp::create(
        builder, loc, mlir::LLVM::LLVMPointerType::get(builder.getContext()),
        type, one);
  }
  return {};
}

auto createLocalSlot(mlir::OpBuilder &builder, mlir::Location loc,
                     mlir::Value initial) -> mlir::Value {
  if (!initial) {
    return {};
  }
  auto slot = createLocalStorage(builder, loc, initial.getType());
  auto address = Address::get(slot, initial.getType());
  if (mlir::failed(address) ||
      mlir::failed(address->store(builder, loc, initial))) {
    return {};
  }
  return slot;
}

} // namespace clang_mlir
