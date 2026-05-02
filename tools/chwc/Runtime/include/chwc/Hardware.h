#ifndef CHWC_RUNTIME_HARDWARE_H
#define CHWC_RUNTIME_HARDWARE_H

#include <cstddef>
#include <type_traits>

namespace chwc {

class Hardware {
public:
  virtual ~Hardware() = default;
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
