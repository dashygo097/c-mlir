// RUN: chwc %s -module=SignedMinMax | FileCheck %s

#include <chwc/Runtime.h>

class SignedMinMax final : public Module {
public:
  Input<SInt<16>> a;
  Input<SInt<16>> b;

  Output<SInt<16>> min_out;
  Output<SInt<16>> max_out;
  Output<Bool> a_lt_b;
  Output<Bool> a_ge_b;

  HW_RESET void rst() {}

  HW_CLOCK_TICK void tick() {
    a_lt_b = a < b;
    a_ge_b = a >= b;

    min_out = b;
    max_out = a;

    if (a < b) {
      min_out = a;
      max_out = b;
    }
  }
};

// CHECK-LABEL: hw.module @SignedMinMax(in %clk : !seq.clock, in %rst : i1, in %a : i16, in %b : i16, out min_out : i16, out max_out : i16, out a_lt_b : i1, out a_ge_b : i1)
// CHECK-NOT: arith.
// CHECK: comb.icmp slt %a, %b : i16
// CHECK: comb.icmp sge %a, %b : i16
// CHECK: comb.mux {{.*}} : i16
// CHECK: comb.mux {{.*}} : i16
// CHECK: hw.output {{.*}} : i16, i16, i1, i1
