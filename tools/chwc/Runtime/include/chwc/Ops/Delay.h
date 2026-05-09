#ifndef CHWC_OPS_DELAY_H
#define CHWC_OPS_DELAY_H

#include "chwc/Signal.h"

namespace chwc {

template <typename T> constexpr auto RegNext(T val) -> Reg<T> {
  return Reg<T>(val);
}

template <typename T, ObjectKind Kind>
constexpr auto RegNext(Signal<T, Kind> val) -> Reg<T> {
  return Reg<T>(val);
}

template <std::size_t CycleDelayed, typename T>
constexpr auto Delay(T value) -> Reg<T> {
  static_assert(CycleDelayed >= 1, "Delay cycle count must be >= 1");

  if constexpr (CycleDelayed == 1) {
    return RegNext(value);
  } else {
    return RegNext(Delay<CycleDelayed - 1, T>(value));
  }
}

template <std::size_t CycleDelayed, typename T, ObjectKind Kind>
constexpr auto Delay(Signal<T, Kind> value) -> Reg<T> {
  static_assert(CycleDelayed >= 1, "Delay cycle count must be >= 1");

  if constexpr (CycleDelayed == 1) {
    return RegNext(value);
  } else {
    return RegNext(Delay<CycleDelayed - 1, T, Kind>(value));
  }
}

} // namespace chwc

#endif // CHWC_OPS_DELAY_H
