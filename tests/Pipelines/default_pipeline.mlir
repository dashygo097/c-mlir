// RUN: cmlir-opt %s | FileCheck %s --check-prefix=RAW
// RUN: cmlir-opt %s --pass-pipeline='builtin.module(default-pipeline)' | FileCheck %s --check-prefix=OPT
// RUN: not cmlir-opt %s --pass-pipeline='builtin.module(default-pipeline{unknown-option=true})' 2>&1 | FileCheck %s --check-prefix=OPTION

// The opt driver does not implicitly execute the compiler pipeline.
// RAW-LABEL: func.func @promote_and_fold
// RAW: memref.alloca
// RAW: memref.store
// RAW: memref.load
// RAW: arith.sitofp
// RAW: arith.fptosi
// RAW: return
// OPT-LABEL: func.func @promote_and_fold
// OPT-NOT: memref.
// OPT-NOT: arith.sitofp
// OPT-NOT: arith.fptosi
// OPT: return %arg0 : i32
// OPTION: unknown-option
func.func @promote_and_fold(%x: i32) -> i32 {
  %slot = memref.alloca() : memref<i32>
  memref.store %x, %slot[] : memref<i32>
  %value = memref.load %slot[] : memref<i32>
  %float = arith.sitofp %value : i32 to f64
  %result = arith.fptosi %float : f64 to i32
  return %result : i32
}
