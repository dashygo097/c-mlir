#ifndef CMLIR_PASSES_TD
#define CMLIR_PASSES_TD

include "mlir/Pass/PassBase.td"

def Mem2RegPass 
    : Pass<"cmlir-mem2reg"> {
  let summary = "Promote memory to register.";
  let constructor = "cmlir::createMem2RegPass()";
}

def Raise2AffinePass
    : Pass<"cmlir-raise2affine"> {
  let summary = "Raise scf to affine dialect.";
  let constructor = "cmlir::createRaise2AffinePass()";
}

#endif // CMLIR_PASSES_TD
