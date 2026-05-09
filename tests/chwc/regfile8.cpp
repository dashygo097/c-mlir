// RUN: chwc %s -module=RegFile8 | FileCheck %s

#include <chwc/Runtime.h>

class RegFile8 final : public Module {
public:
  Input<Bool> wen;
  Input<UInt<3>> waddr;
  Input<UInt<3>> raddr0;
  Input<UInt<3>> raddr1;
  Input<UInt<32>> wdata;

  Output<UInt<32>> rdata0;
  Output<UInt<32>> rdata1;

  Reg<UInt<32>> regs[8];

  HW_RESET void rst() {
    for (int i = 0; i < 8; i++) {
      regs[i] = 0;
    }
  }

  HW_CLOCK_TICK void tick() {
    if (wen) {
      regs[waddr] = wdata;
    }

    rdata0 = regs[raddr0];
    rdata1 = regs[raddr1];
  }
};

// CHECK-LABEL: hw.module @RegFile8(in %clk : !seq.clock, in %rst : i1, in %wen : i1, in %waddr : i3, in %raddr0 : i3, in %raddr1 : i3, in %wdata : i32, out rdata0 : i32, out rdata1 : i32)
// CHECK-NOT: arith.
// CHECK: %[[RESET:.+]] = hw.aggregate_constant
// CHECK-SAME: !hw.array<8xi32>
// CHECK: %[[REGS:.+]] = seq.firreg %[[NEXT:.+]] clock %clk reset sync %rst, %[[RESET]] : !hw.array<8xi32>
// CHECK: %[[INJECT:.+]] = hw.array_inject %[[REGS]][%waddr], %wdata : !hw.array<8xi32>, i3
// CHECK: %[[NEXT]] = comb.mux %wen, %[[INJECT]], %[[REGS]] : !hw.array<8xi32>
// CHECK: %[[RDATA0:.+]] = hw.array_get %[[REGS]][%raddr0] : !hw.array<8xi32>, i3
// CHECK: %[[RDATA1:.+]] = hw.array_get %[[REGS]][%raddr1] : !hw.array<8xi32>, i3
// CHECK: hw.output %[[RDATA0]], %[[RDATA1]] : i32, i32
