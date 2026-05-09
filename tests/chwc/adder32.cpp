// RUN: chwc %s -module=Adder32 | FileCheck %s

#include <chwc/Runtime.h>

class Adder32 final : public Module {
public:
  Input<UInt<32>> in1;
  Input<UInt<32>> in2;
  Output<UInt<32>> out;

  HW_RESET void rst() {}

  HW_CLOCK_TICK void tick() { out = in1 + in2; }
};

// CHECK: hw.module @Adder32(in %clk : !seq.clock, in %rst : i1, in %in1 : i32, in %in2 : i32, out out : i32)
// CHECK: %[[V0:.*]] = comb.add %in1, %in2 : i32
// CHECK: hw.output %[[V0]] : i32
