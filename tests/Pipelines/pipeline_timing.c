// RUN: clang-to-mlir %s --pipeline-timing -o %t.mlir -- -xc 2> %t.timing
// RUN: FileCheck %s --check-prefix=TIMING --input-file=%t.timing
// RUN: mlir-opt %t.mlir -o %t.verified.mlir
// RUN: clang-to-mlir %s -o %t.quiet.mlir -- -xc 2> %t.quiet
// RUN: FileCheck %s --check-prefix=QUIET --allow-empty --input-file=%t.quiet
// RUN: clang-to-mlir %s --disable-opt -o %t.raw.mlir -- -xc
// RUN: cmlir-opt %t.raw.mlir --default-pipeline --mlir-timing -o %t.opt.mlir 2> %t.opt.timing
// RUN: FileCheck %s --check-prefix=TIMING --input-file=%t.opt.timing
// RUN: cmlir-opt %t.raw.mlir --default-pipeline -o %t.opt.quiet.mlir 2> %t.opt.quiet
// RUN: FileCheck %s --check-prefix=QUIET --allow-empty --input-file=%t.opt.quiet
// RUN: clang-to-mlir %s --dump-stages -o %t.stages.mlir -- -xc 2> %t.stages
// RUN: FileCheck %s --check-prefix=STAGES --input-file=%t.stages

int pipeline(int value) {
  int local = value;
  local += 1;
  return local;
}

// TIMING: Execution time report
// TIMING-DAG: SROA
// TIMING-DAG: Mem2Reg
// TIMING-DAG: SCCPPass
// TIMING-DAG: ArithFoldCastsPass
// QUIET-NOT: Execution time report
// STAGES: // clang-to-mlir stage: AST lowering
// STAGES: // clang-to-mlir stage: SSA cleanup
// STAGES: // clang-to-mlir stage: loop preparation
// STAGES: // clang-to-mlir stage: loop transformations
// STAGES: // clang-to-mlir stage: final cleanup
