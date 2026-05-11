#ifndef CHWC_RUNTIME_REGINIT_H
#define CHWC_RUNTIME_REGINIT_H

#include "chwc/Signal.h"
#include <cstddef>
#include <type_traits>

namespace chwc {

template <typename T, typename T::storage_type Init = 0>
class RegInit final : public Signal<T, ObjectKind::Reg> {
public:
  using Base = Signal<T, ObjectKind::Reg>;

  using value_type = T;
  using storage_type = typename T::storage_type;
  using index_type = typename T::index_type;

  static constexpr std::size_t width = T::width;
  static constexpr bool is_signed = T::is_signed;
  static constexpr ObjectKind kind = ObjectKind::Reg;
  static constexpr storage_type init_value = Init;

  constexpr RegInit() : Base(T(Init)) {}

  constexpr RegInit(const T &value) : Base(value) {}

  template <typename U, typename = std::enable_if_t<
                            !std::is_same_v<RegInit<T, Init>, std::decay_t<U>>>>
  constexpr RegInit(const U &value) : Base(T(value)) {}

  using Base::operator=;
};

template <typename T, typename T::storage_type Init>
struct TypeTraits<RegInit<T, Init>> {
  static constexpr bool is_chwc_type = true;
  static constexpr bool is_signal = true;
  static constexpr bool is_signed = T::is_signed;
  static constexpr std::size_t width = T::width;
  static constexpr ObjectKind kind = ObjectKind::Reg;
  static constexpr typename T::storage_type init_value = Init;

  using value_type = T;
  using storage_type = typename T::storage_type;
  using index_type = typename T::index_type;
};

} // namespace chwc

#endif
