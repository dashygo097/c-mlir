#ifndef CMLIR_PASSES_TD
#define CMLIR_PASSES_TD

include "mlir/Pass/PassBase.td"

// Common optimization passes.
def ArithCastPropPass
    : Pass<"arith-cast-prop"> {
  let summary = "Eliminate provably exact arithmetic cast round trips.";
  let constructor = "cmlir::createArithCastPropPass()";
  let dependentDialects = ["mlir::arith::ArithDialect"];
  let description = [{
    Eliminate matching signed or unsigned integer-to-float-to-integer casts
    only when the float precision and exponent range can represent every
    source integer exactly. Arithmetic is not moved across these casts.
    Matching index cast round trips are removed only when the data layout
    or inferred integer range proves that no bits can be lost.
  }];
}

// Explicit semantic policy, used only when the caller permits contraction.
def EnableFPContractPass : Pass<"enable-fp-contract"> {
  let summary = "Permit floating-point contraction on arith addf and mulf.";
  let constructor = "cmlir::createEnableFPContractPass()";
  let dependentDialects = ["mlir::arith::ArithDialect"];
  let description = [{
    Add the contract fast-math flag without changing other fast-math flags.
    This explicitly relaxes floating-point semantics; it is not enabled by
    default and does not permit reassociation.
  }];
}

// Loop optimization passes.
def LoopUnrollPass
    : Pass<"loop-unroll"> {
  let summary = "Unroll loops.";
  let constructor = "cmlir::createLoopUnrollPass()";
  let dependentDialects = ["mlir::arith::ArithDialect", "mlir::scf::SCFDialect"];
  let description = [{
    Interpret CMLIR unroll hints and delegate full or partial unrolling to the
    upstream SCF loop utilities. Loops without unroll hints are unchanged.
  }];
}

def LoopVectorizePass
    : Pass<"loop-vectorize"> {
  let summary = "Vectorize loops.";
  let constructor = "cmlir::createLoopVectorizePass()";
  let description = [{
    Interpret CMLIR vectorization hints, raise eligible SCF loops with upstream
    conversion patterns, and use upstream Affine dependence analysis and
    vectorization utilities. Floating-point reductions require reassociation
    permission. Lower the temporary Affine representation back to SCF.
  }];
}

// Representation conversion passes (optional).
def Struct2MemrefPass
    : Pass<"struct2memref"> {
  let summary = "Convert struct types to memref types.";
  let constructor = "cmlir::createStruct2MemrefPass()";
  let dependentDialects = ["mlir::arith::ArithDialect", "mlir::memref::MemRefDialect"];
  let description = [{
    This pass converts struct types to memref types by performing the following steps:
    1. Identify all struct types in the IR and determine their fields and sizes.
    2. Replace struct type definitions with equivalent memref type definitions that can hold the same data.
    3. Update all operations that use the original struct types to use the new memref types, ensuring that data access patterns are preserved.
  }];
}

#endif // CMLIR_PASSES_TD
