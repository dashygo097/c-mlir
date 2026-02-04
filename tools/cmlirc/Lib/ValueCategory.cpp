//===- ValueCategory.cc ------------------------------------------*- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "./ValueCategory.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/OpDefinition.h"

namespace cmlirc {
using namespace mlir;
using namespace mlir::arith;

ValueCategory::ValueCategory(Value val, bool isReference)
    : val(val), isReference(isReference) {
  assert(val && "null value");

  if (isReference) {
    if (!(isa<MemRefType>(val.getType()) ||
          isa<LLVM::LLVMPointerType>(val.getType()))) {
      llvm::errs() << "val: " << val << "\n";
    }
    assert((isa<MemRefType>(val.getType()) ||
            isa<LLVM::LLVMPointerType>(val.getType())) &&
           "Reference value must have pointer/memref type");
  }
}

Value ValueCategory::getValue(Location loc, OpBuilder &builder) const {
  assert(val && "must be not-null");

  if (!isReference)
    return val;
  if (isa<LLVM::LLVMPointerType>(val.getType())) {
    return LLVM::LoadOp::create(builder, loc, builder.getI32Type(), val);
  }
  if (auto mt = dyn_cast<MemRefType>(val.getType())) {
    assert(mt.getShape().size() == 1 && "must have shape 1");
    auto c0 = ConstantIndexOp::create(builder, loc, 0);
    return memref::LoadOp::create(builder, loc, val, std::vector<Value>({c0}));
  }

  llvm_unreachable("type must be LLVMPointer or MemRef");
}

void ValueCategory::store(Location loc, OpBuilder &builder,
                          Value toStore) const {
  assert(isReference && "must be a reference");
  assert(val && "expect not-null");

  if (isa<LLVM::LLVMPointerType>(val.getType())) {
    LLVM::StoreOp::create(builder, loc, toStore, val);
    return;
  }

  if (auto mt = dyn_cast<MemRefType>(val.getType())) {
    assert(mt.getShape().size() == 1 && "must have size 1");

    if (isa<LLVM::LLVMPointerType>(toStore.getType())) {
      llvm::errs() << "Warning: Converting pointer to memref\n";
    }

    auto c0 = ConstantIndexOp::create(builder, loc, 0);
    memref::StoreOp::create(builder, loc, toStore, val,
                            std::vector<Value>({c0}));
    return;
  }

  llvm_unreachable("type must be LLVMPointer or MemRef");
}

ValueCategory ValueCategory::dereference(Location loc,
                                         OpBuilder &builder) const {
  assert(val && "val must be not-null");

  if (isa<LLVM::LLVMPointerType>(val.getType())) {
    if (!isReference)
      return ValueCategory(val, /*isReference*/ true);
    else
      return ValueCategory(
          LLVM::LoadOp::create(builder, loc, builder.getI32Type(), val),
          /*isReference*/ true);
  }

  if (auto mt = dyn_cast<MemRefType>(val.getType())) {
    auto c0 = ConstantIndexOp::create(builder, loc, 0);
    auto shape = std::vector<int64_t>(mt.getShape());

    if (isReference) {
      if (shape.size() > 1) {
        shape.erase(shape.begin());
        // auto mt0 = MemRefType::get(shape, mt.getElementType(),
        // mt.getLayout(),
        //                            mt.getMemorySpace());
        llvm::errs()
            << "Warning: SubIndexOp not available, using alternative\n";
        return ValueCategory(val, /*isReference*/ true);
      } else {
        return ValueCategory(
            memref::LoadOp::create(builder, loc, val, std::vector<Value>({c0})),
            /*isReference*/ true);
      }
    }
    return ValueCategory(val, /*isReference*/ true);
  }

  llvm_unreachable("type must be LLVMPointer or MemRef");
}

} // namespace cmlirc
