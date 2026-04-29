#ifndef CHWC_RUNTIME_HARDWARE_H
#define CHWC_RUNTIME_HARDWARE_H

#include <cstddef>
#include <type_traits>

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

namespace chwc {

class Hardware {
public:
  virtual ~Hardware() = default;

  virtual void reset() {}
  virtual void clock_tick() {}

  [[nodiscard]] virtual auto name() const noexcept -> const char * {
    return "Unknown Hardware";
  }
};

template <typename T>
inline auto reset(T &hardware)
    -> std::enable_if_t<std::is_base_of<Hardware, T>::value, void> {
  hardware.reset();
}

template <typename T>
inline auto clock_tick(T &hardware)
    -> std::enable_if_t<std::is_base_of<Hardware, T>::value, void> {
  hardware.clock_tick();
}

template <typename T>
inline auto step(T &hardware)
    -> std::enable_if_t<std::is_base_of<Hardware, T>::value, void> {
  hardware.clock_tick();
}

template <typename T>
inline auto run(T &hardware, std::size_t cycles)
    -> std::enable_if_t<std::is_base_of<Hardware, T>::value, void> {
  for (std::size_t i = 0; i < cycles; ++i) {
    hardware.clock_tick();
  }
}

template <typename T>
inline auto reset_and_run(T &hardware, std::size_t cycles)
    -> std::enable_if_t<std::is_base_of<Hardware, T>::value, void> {
  hardware.reset();
  run(hardware, cycles);
}

} // namespace chwc

#endif // CHWC_RUNTIME_HARDWARE_H
