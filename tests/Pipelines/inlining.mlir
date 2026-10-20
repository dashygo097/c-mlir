// RUN: cmlir-opt %s --default-pipeline | FileCheck %s --check-prefix=CALL
// RUN: cmlir-opt %s --pass-pipeline='builtin.module(default-pipeline{inline=true})' | FileCheck %s --check-prefix=INLINE

// CALL-LABEL: func.func @caller
// CALL: call @increment
// INLINE-NOT: func.func private @increment
// INLINE-LABEL: func.func @caller
// INLINE-NOT: call
// INLINE: arith.addi
// INLINE: return
func.func private @increment(%x: i32) -> i32 {
  %one = arith.constant 1 : i32
  %sum = arith.addi %x, %one : i32
  return %sum : i32
}

func.func @caller(%x: i32) -> i32 {
  %result = func.call @increment(%x) : (i32) -> i32
  return %result : i32
}
