// RUN: chwc %s -module=Counter | FileCheck %s
#include <chwc/Runtime.h>

class Counter final : public Hardware {
public:
  Input<UInt<1>> en;
  Output<UInt<16>> out;
  Reg<UInt<16>> value;

  HW_RESET void rst() { value = 0; }

  HW_CLOCK_TICK void tick() {
    if (en) {
      value = add_one(value);
    }
    out = value;
  }

  HW_FUNC UInt<16> add_one(UInt<16> input) { return input + 1; }
};

// CHECK: hw.module @Counter(in %clk : !seq.clock, in %rst : i1, in %en : i1, out out : i16)
// CHECK-DAG: %[[c1_i16:.+]] = hw.constant 1 : i16
// CHECK-DAG: %[[c0_i16:.+]] = hw.constant 0 : i16
// CHECK-DAG: %[[V0:.*]] = comb.add %{{.*}}, %[[c1_i16]] : i16
// CHECK-DAG: %[[V1:.*]] = comb.mux %en, %[[V0]], %{{.*}} : i16
// CHECK-DAG: %[[value:.*]] = seq.firreg %[[V1]] clock %clk reset sync %rst, %[[c0_i16]] : i16
// CHECK: hw.output %[[value]] : i16
