#ifndef CMLIR_PASSES_TD
#define CMLIR_PASSES_TD

include "mlir/Pass/PassBase.td"

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

def LoopUnrollPass
    : Pass<"cmlir-loop-unroll"> {
  let summary = "Unroll loops.";
  let constructor = "cmlir::createLoopUnrollPass()";
  let description = [{
    This pass unrolls loops by performing the following steps:
    1. Identify loops that are suitable for unrolling based on their iteration counts and body complexity.
    2. Duplicate the loop body multiple times according to the unroll factor, adjusting loop control variables and conditions accordingly.
    3. Remove the original loop construct if it is fully unrolled, or update it to reflect the reduced iteration count if it is partially unrolled.
  }];
}

def LoopVectorizePass
    : Pass<"cmlir-loop-vectorize"> {
  let summary = "Vectorize loops.";
  let constructor = "cmlir::createLoopVectorizePass()";
  let description = [{
    This pass vectorizes loops by performing the following steps:
    1. Identify loops that are suitable for vectorization based on their iteration counts, data dependencies, and memory access patterns.
    2. Transform the loop body to operate on vectors of data instead of scalar values, which may involve changing the types of variables and operations to their vector equivalents.
    3. Adjust loop control variables and conditions to reflect the new vectorized structure, ensuring that the semantics of the original loop are preserved while enabling improved performance through SIMD execution.
  }];
}

def ArithCastPropPass
    : Pass<"cmlir-arith-cast-prop"> {
  let summary = "Perform arith casts propgation";
  let constructor = "cmlir::createArithCastPropPass()";
  let description = [{
    This pass propgates the arith cast from float -> int or int -> float and fold arith.index_cast for readability.
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

 def RaiseMemref2AffinePass
    : Pass<"cmlir-raise-memref2affine"> {
  let summary = "Raise standard memref memory accesses to affine operations.";
  let constructor = "cmlir::createRaiseMemref2AffinePass()";
  let description = [{
    This pass raises standard memref accesses to affine operations by performing the following steps:
    1. Identify standard `memref.load` and `memref.store` operations that use index expressions to access memory.
    2. Analyze the access indices to determine if they can be safely represented as valid affine maps (i.e., linear combinations of surrounding loop induction variables and symbols).
    3. Replace the identified memref operations with equivalent `affine.load` and `affine.store` operations, preserving the original memory semantics while unlocking advanced polyhedral loop transformations and optimizations (such as tiling, fusion, and unrolling) specific to the Affine dialect.
  }];
}

def RaiseSCF2AffinePass
    : Pass<"cmlir-raise-scf2affine"> {
  let summary = "Raise scf operations to affine operations.";
  let constructor = "cmlir::createRaiseSCF2AffinePass()";
  let description = [{
    This pass raises scf operations to affine operations by performing the following steps:
    1. Identify scf operations that can be represented as affine operations, such as loops and conditionals.
    2. Analyze the control flow and data dependencies of these scf operations to determine if they can be safely transformed into affine constructs.
    3. Replace the identified scf operations with equivalent affine operations, ensuring that the semantics of the original code are preserved while enabling further optimizations that are specific to affine constructs.
  }];
}

#endif // CMLIR_PASSES_TD
