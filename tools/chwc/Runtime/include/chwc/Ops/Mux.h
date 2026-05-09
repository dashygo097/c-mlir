#ifndef CHWC_OPS_MUX_H
#define CHWC_OPS_MUX_H

#include "chwc/Types/Enum.h"

namespace chwc {

template <typename T>
constexpr auto Mux(Bool cond, T true_val, T false_val) -> T {
  return static_cast<bool>(cond) ? true_val : false_val;
}

template <typename T, ObjectKind Kind>
constexpr auto Mux(Bool cond, Signal<T, Kind> true_val,
                   Signal<T, Kind> false_val) -> Signal<T, Kind> {
  return static_cast<bool>(cond) ? true_val : false_val;
}

template <typename T, ObjectKind TrueKind, ObjectKind FalseKind>
constexpr auto Mux(Bool cond, Signal<T, TrueKind> true_val,
                   Signal<T, FalseKind> false_val) -> T {
  return static_cast<bool>(cond) ? true_val : false_val;
}

} // namespace chwc

#endif // CHWC_OPS_MUX_H
