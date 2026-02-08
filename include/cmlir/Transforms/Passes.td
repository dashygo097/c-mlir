#ifndef CMLIR_PASSES_TD
#define CMLIR_PASSES_TD

include "mlir/Pass/PassBase.td"

def ConstantMergePass
    : Pass<"cmlir-constant-merge", "mlir::func::FuncOp"> {
  let summary = "Merge duplicate constant operations";
  let description = [{}];

  let constructor = "cmlir::createConstantMergePass()";

  let dependentDialects = ["mlir::arith::ArithDialect"];
}

#endif // CMLIR_PASSES_TD
