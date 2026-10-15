// RUN: clang-to-mlir %s --pipeline-timing -o %t.mlir -- -xc 2> %t.timing
// RUN: FileCheck %s --check-prefix=TIMING --input-file=%t.timing
// RUN: mlir-opt %t.mlir -o %t.verified.mlir
// RUN: clang-to-mlir %s -o %t.quiet.mlir -- -xc 2> %t.quiet
// RUN: FileCheck %s --check-prefix=QUIET --allow-empty --input-file=%t.quiet

int pipeline(int value) {
  int local = value;
  local += 1;
  return local;
}

// TIMING: Execution time report
// TIMING-DAG: SROA
// TIMING-DAG: Mem2Reg
// TIMING-DAG: SCCPPass
// TIMING-DAG: ArithCastPropPass
// QUIET-NOT: Execution time report
