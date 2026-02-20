#ifndef CMLIR_PASSES_TD
#define CMLIR_PASSES_TD

include "mlir/Pass/PassBase.td"

def Mem2RegPass 
    : Pass<"cmlir-mem2reg"> {
  let summary = "Promote memory to register.";
  let constructor = "cmlir::createMem2RegPass()";

  let description = [{
    This pass promotes memory to register by performing the following steps:
    1. Identify all memory allocations (e.g., `alloca` operations) that can be promoted.
    2. Replace loads and stores to these memory locations with direct register accesses.
    3. Remove the original memory allocations and associated load/store operations.
  }];
}

def ConstPropPass
    : Pass<"cmlir-constprop"> {
  let summary = "Perform constant propagation.";
  let constructor = "cmlir::createConstPropPass()";

  let description = [{
    This pass performs constant propagation for the following patterns:
    1. affine.store(cast(const)) -> affine.store(const)
    2. memref.store(cast(const)) -> memref.store(const)
  }];
}

def FMAPass
    : Pass<"cmlir-fma"> {
  let summary = "Fuse multiply-add operations.";
  let constructor = "cmlir::createFMAPass()";

  let description = [{
    This pass identifies patterns of the form `a * b + c` and fuses them into a single fused multiply-add (FMA) operation,
    which can improve performance by reducing the number of separate instructions and enabling better use of hardware capabilities.
    1. arith.add(arith.mul(a, b), c) -> math.fma(a, b, c)
  }];
}

def Struct2MemrefPass
    : Pass<"cmlir-struct2memref"> {
  let summary = "Convert struct types to memref types.";
  let constructor = "cmlir::createStruct2MemrefPass()";
  let description = [{
    This pass converts struct types to memref types by performing the following steps:
    1. Identify all struct types in the IR and determine their fields and sizes.
    2. Replace struct type definitions with equivalent memref type definitions that can hold the same data.
    3. Update all operations that use the original struct types to use the new memref types, ensuring that data access patterns are preserved.
  }];
}

#endif // CMLIR_PASSES_TD
