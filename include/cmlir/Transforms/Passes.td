#ifndef CMLIR_PASSES_TD
#define CMLIR_PASSES_TD

include "mlir/Pass/PassBase.td"

def Mem2RegPass 
    : Pass<"cmlir-mem2reg"> {
  let summary = "Promote memory to register.";
  let constructor = "cmlir::createMem2RegPass()";
}

#endif // CMLIR_PASSES_TD
