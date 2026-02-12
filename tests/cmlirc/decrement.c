// RUN: split-file %s %t

// RUN: cmlirc %t/pre_decrement.c | FileCheck %s --check-prefix=CHECKPRE
// RUN: cmlirc %t/post_decrement.c --canonicalize | FileCheck %s --check-prefix=CHECKPOST

//--- pre_decrement.c
int pre_decrement(int a) { return --a; }

// CHECKPRE: func.func @pre_decrement(%arg0: i32) -> i32
// CHECKPRE-DAG: %[[c1_i32:.+]] = arith.constant 1 : i32
// CHECKPRE: %[[V0:.*]] = arith.subi %arg0, %c1_i32 : i32
// CHECKPRE: return %[[V0]] : i32

//--- post_decrement.c
int post_decrement(int a) { return a--; }

// CHECKPOST: func.func @post_decrement(%arg0: i32) -> i32
// CHECKPOST: return %arg0 : i32
