// RUN: chwc %s -module=AddTwo | FileCheck %s

#include <chwc/Runtime.h>

class AddOne : public Module {
public:
  Input<UInt<16>> in;
  Output<UInt<16>> out;

  HW_RESET void rst() {}

  HW_CLOCK_TICK void tick() { out = in + 1; }
};

class AddTwo final : public Module {
public:
  Input<UInt<16>> in;
  Output<UInt<16>> out;

  Instance<AddOne> add_one_inst_0;
  Instance<AddOne> add_one_inst_1;

  HW_RESET void rst() {}

  HW_CLOCK_TICK void tick() {
    add_one_inst_0.io.in = in;
    add_one_inst_1.io.in = add_one_inst_0.io.out;
    out = add_one_inst_1.io.out;
  }
};

// CHECK: hw.module @AddOne(in %clk : !seq.clock, in %rst : i1, in %in : i16, out out : i16)
// CHECK-DAG: %[[c1_i16:.+]] = hw.constant 1 : i16
// CHECK: %[[OUT:.+]] = comb.add %in, %[[c1_i16]] : i16
// CHECK: hw.output %[[OUT]] : i16

// CHECK: hw.module @AddTwo(in %clk : !seq.clock, in %rst : i1, in %in : i16, out out : i16)
// CHECK: %[[OUT1:.+]] = hw.instance "add_one_inst_0" @AddOne(clk: %clk: !seq.clock, rst: %rst: i1, in: %in: i16) -> (out: i16)
// CHECK: %[[OUT2:.+]] = hw.instance "add_one_inst_1" @AddOne(clk: %clk: !seq.clock, rst: %rst: i1, in: %[[OUT1]]: i16) -> (out: i16)
// CHECK: hw.output %[[OUT2]] : i16
