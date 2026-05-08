#ifndef CHWC_ANNOTATE_RUNTIME_H
#define CHWC_ANNOTATE_RUNTIME_H

#if defined(__clang__)
#define CHWC_ANNOTATE(name) __attribute__((annotate(name)))
#define CHWC_METHOD_ANNOTATE(name) [[clang::annotate(name)]]
#else
#define CHWC_ANNOTATE(name)
#define CHWC_METHOD_ANNOTATE(name)
#endif

#define __reset__ CHWC_METHOD_ANNOTATE("hw.reset")
#define __clock_tick__ CHWC_METHOD_ANNOTATE("hw.clock_tick")
#define __func__ CHWC_METHOD_ANNOTATE("hw.func")

#define HW_RESET CHWC_METHOD_ANNOTATE("hw.reset")
#define HW_CLOCK_TICK CHWC_METHOD_ANNOTATE("hw.clock_tick")
#define HW_FUNC CHWC_METHOD_ANNOTATE("hw.func")

#endif // CHWC_ANNOTATE_RUNTIME_H
