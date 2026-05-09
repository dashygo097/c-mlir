// RUN: chwc %s -module=Adder | FileCheck %s

#include <chwc/Runtime.h>

template <int WIDTH>
class Adder final : public Module {
public:
  Input<UInt<WIDTH>> in1;
  Input<UInt<WIDTH>> in2;
  Output<UInt<WIDTH>> out;

  __reset__ void rst() {}

  __clock_tick__ void tick() {
    out = in1 + in2;
  }
};

// CHECK: hw.module @Adder<WIDTH: i32>(in %clk : !seq.clock, in %rst : i1, in %in1 : !hw.int<#hw.param.decl.ref<"WIDTH">>, in %in2 : !hw.int<#hw.param.decl.ref<"WIDTH">>, out out : !hw.int<#hw.param.decl.ref<"WIDTH">>)
// CHECK: %[[OUT:.+]] = comb.add %in1, %in2 : !hw.int<#hw.param.decl.ref<"WIDTH">>
// CHECK: hw.output %[[OUT]] : !hw.int<#hw.param.decl.ref<"WIDTH">>
