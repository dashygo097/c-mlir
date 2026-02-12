// RUN: split-file %s %t

// RUN: cmlirc %t/pre_increment.c | FileCheck %s --check-prefix=CHECKPRE
// RUN: cmlirc %t/post_increment.c --canonicalize | FileCheck %s --check-prefix=CHECKPOST

//--- pre_increment.c
int pre_increment(int a) { return ++a; }

// CHECKPRE: func.func @pre_increment(%arg0: i32) -> i32
// CHECKPRE-DAG: %[[c1_i32:.+]] = arith.constant 1 : i32
// CHECKPRE: %[[V0:.*]] = arith.addi %arg0, %c1_i32 : i32
// CHECKPRE: return %[[V0]] : i32

//--- post_increment.c
int post_increment(int a) { return a++; }

// CHECKPOST: func.func @post_increment(%arg0: i32) -> i32
// CHECKPOST: return %arg0 : i32
