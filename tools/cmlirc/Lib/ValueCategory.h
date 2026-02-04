//===- ValueCategory.h -------------------------------------------*- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef CLANG_MLIR_VALUE_CATEGORY
#define CLANG_MLIR_VALUE_CATEGORY

#include "mlir/IR/Builders.h"
#include "mlir/IR/Value.h"

namespace cmlirc {
using namespace mlir;

// Represents a rhs or lhs value.
class ValueCategory {
public:
  Value val;
  bool isReference;

public:
  ValueCategory() : val(nullptr), isReference(false) {};
  ValueCategory(std::nullptr_t) : val(nullptr), isReference(false) {};
  ValueCategory(Value val, bool isReference);

  Value getValue(Location loc, OpBuilder &builder) const;
  void store(Location loc, OpBuilder &builder, Value toStore) const;
  ValueCategory dereference(Location loc, OpBuilder &builder) const;
};

} // namespace cmlirc

#endif
