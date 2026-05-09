// RUN: chwc %s -module=Mux2 | FileCheck %s

#include <chwc/Runtime.h>

class Mux2 final : public Module {
public:
  Input<Bool> sel;
  Input<UInt<8>> a;
  Input<UInt<8>> b;
  Output<UInt<8>> out;

  Reg<UInt<8>> value;

  HW_RESET void rst() { value = 0; }

  HW_CLOCK_TICK void tick() {
    value = Mux(sel, a, b);
    out = value;
  }
};

// CHECK-LABEL: hw.module @Mux2(in %clk : !seq.clock, in %rst : i1, in %sel : i1, in %a : i8, in %b : i8, out out : i8)
// CHECK-NOT: arith.
// CHECK: seq.firreg {{.*}} : i8
// CHECK: comb.mux {{.*}} : i8
// CHECK: hw.output {{.*}} : i8
