// RUN: chwc %s -module=Counter | FileCheck %s

#include <chwc/Runtime.h>

class Counter final : public Module {
public:
  Input<Bool> en;
  Output<UInt<16>> out;
  Reg<UInt<16>> value;

  HW_RESET void rst() { value = 0; }

  HW_CLOCK_TICK void tick() {
    if (en) {
      value = step(value);
    }
    out = value;
  }

  HW_FUNC UInt<16> step(UInt<16> input) { return input + 1; }
};

// CHECK: hw.module @Counter(in %clk : !seq.clock, in %rst : i1, in %en : i1, out out : i16)
// CHECK-DAG: %[[c1_i16:.+]] = hw.constant 1 : i16
// CHECK-DAG: %[[c0_i16:.+]] = hw.constant 0 : i16
// CHECK: %[[VALUE:.+]] = seq.firreg %[[NEXT:.+]] clock %clk reset sync %rst, %[[c0_i16]] : i16
// CHECK: %[[ADD:.+]] = comb.add %[[VALUE]], %[[c1_i16]] : i1
// CHECK: %[[NEXT]] = comb.mux %en, %[[ADD]], %[[VALUE]] : i16
// CHECK: hw.output %[[VALUE]] : i16
