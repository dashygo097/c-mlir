// RUN: cmlir-opt %s --default-pipeline | FileCheck %s --check-prefix=LIFT
// RUN: cmlir-opt %s --pass-pipeline='builtin.module(default-pipeline{lift-cf-to-scf=false})' | FileCheck %s --check-prefix=CFG

func.func private @left(i32) -> i32
func.func private @right(i32) -> i32

// LIFT-LABEL: func.func @choose
// LIFT-NOT: cf.br
// LIFT-NOT: cf.cond_br
// LIFT: scf.if
// LIFT: call @left
// LIFT: } else {
// LIFT: call @right
// LIFT-NOT: cf.br
// LIFT-NOT: cf.cond_br
// LIFT: return
// CFG-LABEL: func.func @choose
// CFG: cf.cond_br
// CFG: call @left
// CFG: call @right
func.func @choose(%condition: i1, %x: i32) -> i32 {
  cf.cond_br %condition, ^left, ^right
^left:
  %a = func.call @left(%x) : (i32) -> i32
  return %a : i32
^right:
  %b = func.call @right(%x) : (i32) -> i32
  return %b : i32
}
