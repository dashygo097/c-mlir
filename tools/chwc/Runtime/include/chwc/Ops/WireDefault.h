#ifndef CHWC_RUNTIME_OPS_WIRE_DEFAULT_H
#define CHWC_RUNTIME_OPS_WIRE_DEFAULT_H

#include "chwc/Signal.h"

namespace chwc {

template <typename T> constexpr auto WireDefault(T value) -> Wire<T> {
  return Wire<T>(value);
}

template <typename T, ObjectKind Kind>
constexpr auto WireDefault(Signal<T, Kind> value) -> Wire<T> {
  return Wire<T>(value);
}

} // namespace chwc

#endif // CHWC_RUNTIME_OPS_WIRE_DEFAULT_H
