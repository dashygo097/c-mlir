#ifndef CHWC_RUNTIME_UINT_H
#define CHWC_RUNTIME_UINT_H

#include "chwc/Types/Signal.h"
#include <cstdint>
#include <limits>

namespace chwc {
template <std::size_t Width> class UInt {
public:
  static_assert(Width >= 1, "UInt width must be positive");
  static_assert(Width <= 64, "UInt currently supports width <= 64");

  using storage_type = std::uint64_t;

  static constexpr std::size_t width = Width;
  static constexpr bool is_signed = false;
  static constexpr ObjectKind kind = ObjectKind::Value;

  constexpr UInt() = default;

  template <typename T, typename = std::enable_if_t<std::is_integral_v<T>>>
  constexpr explicit UInt(T value)
      : value_(normalize(static_cast<storage_type>(value))) {}

  template <std::size_t OtherWidth>
  constexpr explicit UInt(UInt<OtherWidth> value)
      : value_(normalize(static_cast<storage_type>(value.raw()))) {}

  [[nodiscard]] constexpr auto raw() const -> storage_type { return value_; }

  [[nodiscard]] constexpr auto value() const -> storage_type { return value_; }

  constexpr explicit operator bool() const { return value_ != 0; }

  // NOTE: explicit removed for signal unwarpping and debugging
  constexpr operator storage_type() const { return value_; }

  template <typename T, typename = std::enable_if_t<std::is_integral_v<T>>>
  constexpr auto operator=(T value) -> UInt & {
    value_ = normalize(static_cast<storage_type>(value));
    return *this;
  }

  template <std::size_t OtherWidth>
  constexpr auto operator=(UInt<OtherWidth> value) -> UInt & {
    value_ = normalize(value.raw());
    return *this;
  }

  constexpr auto operator+() const -> UInt { return *this; }

  constexpr auto operator-() const -> UInt { return UInt(0 - value_); }

  constexpr auto operator~() const -> UInt { return UInt(~value_); }

  constexpr auto operator!() const -> UInt {
    return UInt<1>(!static_cast<bool>(value_));
  }

  template <typename T, typename = std::enable_if_t<std::is_integral_v<T>>>
  constexpr auto operator+(T rhs) const -> UInt {
    return *this + UInt(rhs);
  }

  template <typename T, typename = std::enable_if_t<std::is_integral_v<T>>>
  constexpr auto operator-(T rhs) const -> UInt {
    return *this - UInt(rhs);
  }

  template <typename T, typename = std::enable_if_t<std::is_integral_v<T>>>
  constexpr auto operator*(T rhs) const -> UInt {
    return *this * UInt(rhs);
  }

  template <typename T, typename = std::enable_if_t<std::is_integral_v<T>>>
  constexpr auto operator/(T rhs) const -> UInt {
    return *this / UInt(rhs);
  }

  template <typename T, typename = std::enable_if_t<std::is_integral_v<T>>>
  constexpr auto operator%(T rhs) const -> UInt {
    return *this % UInt(rhs);
  }

  template <typename T, typename = std::enable_if_t<std::is_integral_v<T>>>
  constexpr auto operator&(T rhs) const -> UInt {
    return *this & UInt(rhs);
  }

  template <typename T, typename = std::enable_if_t<std::is_integral_v<T>>>
  constexpr auto operator|(T rhs) const -> UInt {
    return *this | UInt(rhs);
  }

  template <typename T, typename = std::enable_if_t<std::is_integral_v<T>>>
  constexpr auto operator^(T rhs) const -> UInt {
    return *this ^ UInt(rhs);
  }

  template <typename T, typename = std::enable_if_t<std::is_integral_v<T>>>
  constexpr auto operator<<(T rhs) const -> UInt {
    return *this << UInt(rhs);
  }

  template <typename T, typename = std::enable_if_t<std::is_integral_v<T>>>
  constexpr auto operator>>(T rhs) const -> UInt {
    return *this >> UInt(rhs);
  }

  template <typename T, typename = std::enable_if_t<std::is_integral_v<T>>>
  constexpr auto operator&&(T rhs) const -> UInt {
    return UInt<1>(static_cast<bool>(value_) && static_cast<bool>(rhs));
  }

  template <typename T, typename = std::enable_if_t<std::is_integral_v<T>>>
  constexpr auto operator||(T rhs) const -> UInt {
    return UInt<1>(static_cast<bool>(value_) || static_cast<bool>(rhs));
  }

  template <std::size_t OtherWidth>
  constexpr auto operator+(UInt<OtherWidth> rhs) const -> UInt {
    return UInt(value_ + rhs.raw());
  }

  template <std::size_t OtherWidth>
  constexpr auto operator-(UInt<OtherWidth> rhs) const -> UInt {
    return UInt(value_ - rhs.raw());
  }

  template <std::size_t OtherWidth>
  constexpr auto operator*(UInt<OtherWidth> rhs) const -> UInt {
    return UInt(value_ * rhs.raw());
  }

  template <std::size_t OtherWidth>
  constexpr auto operator/(UInt<OtherWidth> rhs) const -> UInt {
    return rhs.raw() == 0 ? UInt(0) : UInt(value_ / rhs.raw());
  }

  template <std::size_t OtherWidth>
  constexpr auto operator%(UInt<OtherWidth> rhs) const -> UInt {
    return rhs.raw() == 0 ? UInt(0) : UInt(value_ % rhs.raw());
  }

  template <std::size_t OtherWidth>
  constexpr auto operator&(UInt<OtherWidth> rhs) const -> UInt {
    return UInt(value_ & rhs.raw());
  }

  template <std::size_t OtherWidth>
  constexpr auto operator|(UInt<OtherWidth> rhs) const -> UInt {
    return UInt(value_ | rhs.raw());
  }

  template <std::size_t OtherWidth>
  constexpr auto operator^(UInt<OtherWidth> rhs) const -> UInt {
    return UInt(value_ ^ rhs.raw());
  }

  template <std::size_t OtherWidth>
  constexpr auto operator<<(UInt<OtherWidth> rhs) const -> UInt {
    return UInt(value_ << rhs.raw());
  }

  template <std::size_t OtherWidth>
  constexpr auto operator>>(UInt<OtherWidth> rhs) const -> UInt {
    return UInt(value_ >> rhs.raw());
  }

  template <std::size_t OtherWidth>
  constexpr auto operator+=(UInt<OtherWidth> rhs) -> UInt & {
    value_ = normalize(value_ + rhs.raw());
    return *this;
  }

  template <std::size_t OtherWidth>
  constexpr auto operator-=(UInt<OtherWidth> rhs) -> UInt & {
    value_ = normalize(value_ - rhs.raw());
    return *this;
  }

  template <std::size_t OtherWidth>
  constexpr auto operator*=(UInt<OtherWidth> rhs) -> UInt & {
    value_ = normalize(value_ * rhs.raw());
    return *this;
  }

  template <std::size_t OtherWidth>
  constexpr auto operator/=(UInt<OtherWidth> rhs) -> UInt & {
    value_ = rhs.raw() == 0 ? 0 : normalize(value_ / rhs.raw());
    return *this;
  }

  template <std::size_t OtherWidth>
  constexpr auto operator%=(UInt<OtherWidth> rhs) -> UInt & {
    value_ = rhs.raw() == 0 ? 0 : normalize(value_ % rhs.raw());
    return *this;
  }

  template <std::size_t OtherWidth>
  constexpr auto operator&=(UInt<OtherWidth> rhs) -> UInt & {
    value_ = normalize(value_ & rhs.raw());
    return *this;
  }

  template <std::size_t OtherWidth>
  constexpr auto operator|=(UInt<OtherWidth> rhs) -> UInt & {
    value_ = normalize(value_ | rhs.raw());
    return *this;
  }

  template <std::size_t OtherWidth>
  constexpr auto operator^=(UInt<OtherWidth> rhs) -> UInt & {
    value_ = normalize(value_ ^ rhs.raw());
    return *this;
  }

  template <std::size_t OtherWidth>
  constexpr auto operator<<=(UInt<OtherWidth> rhs) -> UInt & {
    value_ = normalize(value_ << rhs.raw());
    return *this;
  }

  template <std::size_t OtherWidth>
  constexpr auto operator>>=(UInt<OtherWidth> rhs) -> UInt & {
    value_ = normalize(value_ >> rhs.raw());
    return *this;
  }

  template <std::size_t OtherWidth>
  constexpr auto operator==(UInt<OtherWidth> rhs) const -> bool {
    return value_ == rhs.raw();
  }

  template <std::size_t OtherWidth>
  constexpr auto operator!=(UInt<OtherWidth> rhs) const -> bool {
    return value_ != rhs.raw();
  }

  template <std::size_t OtherWidth>
  constexpr auto operator<(UInt<OtherWidth> rhs) const -> bool {
    return value_ < rhs.raw();
  }

  template <std::size_t OtherWidth>
  constexpr auto operator<=(UInt<OtherWidth> rhs) const -> bool {
    return value_ <= rhs.raw();
  }

  template <std::size_t OtherWidth>
  constexpr auto operator>(UInt<OtherWidth> rhs) const -> bool {
    return value_ > rhs.raw();
  }

  template <std::size_t OtherWidth>
  constexpr auto operator>=(UInt<OtherWidth> rhs) const -> bool {
    return value_ >= rhs.raw();
  }

  template <std::size_t OtherWidth>
  constexpr auto operator&&(UInt<OtherWidth> rhs) const -> bool {
    return static_cast<bool>(value_) && static_cast<bool>(rhs.raw());
  }

  template <std::size_t OtherWidth>
  constexpr auto operator||(UInt<OtherWidth> rhs) const -> bool {
    return static_cast<bool>(value_) || static_cast<bool>(rhs.raw());
  }

  constexpr auto operator++() -> UInt & {
    *this += UInt(1);
    return *this;
  }

  constexpr auto operator++(int) -> UInt {
    UInt old = *this;
    ++(*this);
    return old;
  }

  constexpr auto operator--() -> UInt & {
    *this -= UInt(1);
    return *this;
  }

  constexpr auto operator--(int) -> UInt {
    UInt old = *this;
    --(*this);
    return old;
  }

  static constexpr auto mask() -> storage_type {
    if constexpr (Width == 64) {
      return std::numeric_limits<storage_type>::max();
    } else {
      return (storage_type{1} << Width) - 1;
    }
  }

  static constexpr auto normalize(storage_type value) -> storage_type {
    return value & mask();
  }

private:
  storage_type value_{0};
};

template <std::size_t Width> struct TypeTraits<UInt<Width>> {
  static constexpr bool is_chwc_type = true;
  static constexpr bool is_signal = false;
  static constexpr bool is_signed = false;
  static constexpr std::size_t width = Width;
  static constexpr ObjectKind kind = ObjectKind::Value;

  using value_type = UInt<Width>;
};

} // namespace chwc

#endif // CHWC_RUNTIME_UINT_H
