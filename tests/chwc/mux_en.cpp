// RUN: chwc %s -module=MuxEn | FileCheck %s

#include <chwc/Runtime.h>

class MuxEn final : public Module {
public:
  Input<Bool> en;
  Input<Bool> mode;
  Input<UInt<8>> a;
  Input<UInt<8>> b;
  Output<UInt<8>> out;

  Reg<UInt<8>> value;

  HW_RESET void rst() { value = 0; }

  HW_CLOCK_TICK void tick() {
    if (en) {
      if (mode) {
        value = a;
      }

      if (!mode) {
        value = b;
      }
    }

    out = value;
  }
};

// CHECK-LABEL: hw.module @MuxEn(in %clk : !seq.clock, in %rst : i1, in %en : i1, in %mode : i1, in %a : i8, in %b : i8, out out : i8)
// CHECK-NOT: arith.
// CHECK: seq.firreg {{.*}} : i8
// CHECK: comb.mux {{.*}} : i8
// CHECK: comb.mux {{.*}} : i8
// CHECK: hw.output {{.*}} : i8
