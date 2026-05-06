#ifndef CHWC_ANNOTATE_RUNTIME_H
#define CHWC_ANNOTATE_RUNTIME_H

#if defined(__clang__)
#define CHWC_ANNOTATE(name) __attribute__((annotate(name)))
#define CHWC_METHOD_ANNOTATE(name) [[clang::annotate(name)]]
#else
#define CHWC_ANNOTATE(name)
#define CHWC_METHOD_ANNOTATE(name)
#endif

#define HW_INPUT CHWC_ANNOTATE("hw.input")
#define HW_OUTPUT CHWC_ANNOTATE("hw.output")
#define HW_REG CHWC_ANNOTATE("hw.reg")
#define HW_WIRE CHWC_ANNOTATE("hw.wire")

#define HW_RESET CHWC_METHOD_ANNOTATE("hw.reset")
#define HW_CLOCK_TICK CHWC_METHOD_ANNOTATE("hw.clock_tick")
#define HW_FUNC CHWC_METHOD_ANNOTATE("hw.func")

#endif // CHWC_ANNOTATE_RUNTIME_H
