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


#endif // CMLIR_PASSES_TD
