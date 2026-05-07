#ifndef CHWC_RUNTIME_SINT_H
#define CHWC_RUNTIME_SINT_H

#include "chwc/Types/Signal.h"
#include "chwc/Types/UInt.h"
#include <cstdint>

namespace chwc {

template <std::size_t Width> class SInt {
public:
  static_assert(Width >= 1, "SInt width must be positive");
  static_assert(Width <= 64, "SInt currently supports width <= 64");

  using storage_type = std::int64_t;

  static constexpr std::size_t width = Width;
  static constexpr bool is_signed = true;
  static constexpr ObjectKind kind = ObjectKind::Value;

  constexpr SInt() = default;

  template <typename T, typename = std::enable_if_t<std::is_integral_v<T>>>
  constexpr explicit SInt(T value)
      : value_(normalize(static_cast<storage_type>(value))) {}

  template <std::size_t OtherWidth>
  constexpr explicit SInt(SInt<OtherWidth> value)
      : value_(normalize(static_cast<storage_type>(value.raw()))) {}

  [[nodiscard]] constexpr auto raw() const -> storage_type { return value_; }

  [[nodiscard]] constexpr auto value() const -> storage_type { return value_; }

  constexpr explicit operator bool() const { return value_ != 0; }

  constexpr explicit operator storage_type() const { return value_; }

  template <typename T, typename = std::enable_if_t<std::is_integral_v<T>>>
  constexpr auto operator=(T value) -> SInt & {
    value_ = normalize(static_cast<storage_type>(value));
    return *this;
  }

  template <std::size_t OtherWidth>
  constexpr auto operator=(SInt<OtherWidth> value) -> SInt & {
    value_ = normalize(value.raw());
    return *this;
  }

  constexpr auto operator+() const -> SInt { return *this; }

  constexpr auto operator-() const -> SInt { return SInt(0 - value_); }

  constexpr auto operator~() const -> SInt { return SInt(~value_); }

  constexpr auto operator!() const -> SInt {
    return UInt<1>(!static_cast<bool>(value_));
  }

  template <typename T, typename = std::enable_if_t<std::is_integral_v<T>>>
  constexpr auto operator+(T rhs) const -> SInt {
    return *this + SInt(rhs);
  }

  template <typename T, typename = std::enable_if_t<std::is_integral_v<T>>>
  constexpr auto operator-(T rhs) const -> SInt {
    return *this - SInt(rhs);
  }

  template <typename T, typename = std::enable_if_t<std::is_integral_v<T>>>
  constexpr auto operator*(T rhs) const -> SInt {
    return *this * SInt(rhs);
  }

  template <typename T, typename = std::enable_if_t<std::is_integral_v<T>>>
  constexpr auto operator/(T rhs) const -> SInt {
    return *this / SInt(rhs);
  }

  template <typename T, typename = std::enable_if_t<std::is_integral_v<T>>>
  constexpr auto operator%(T rhs) const -> SInt {
    return *this % SInt(rhs);
  }

  template <typename T, typename = std::enable_if_t<std::is_integral_v<T>>>
  constexpr auto operator&(T rhs) const -> SInt {
    return *this & SInt(rhs);
  }

  template <typename T, typename = std::enable_if_t<std::is_integral_v<T>>>
  constexpr auto operator|(T rhs) const -> SInt {
    return *this | SInt(rhs);
  }

  template <typename T, typename = std::enable_if_t<std::is_integral_v<T>>>
  constexpr auto operator^(T rhs) const -> SInt {
    return *this ^ SInt(rhs);
  }

  template <typename T, typename = std::enable_if_t<std::is_integral_v<T>>>
  constexpr auto operator<<(T rhs) const -> SInt {
    return *this << SInt(rhs);
  }

  template <typename T, typename = std::enable_if_t<std::is_integral_v<T>>>
  constexpr auto operator>>(T rhs) const -> SInt {
    return *this >> SInt(rhs);
  }

  template <typename T, typename = std::enable_if_t<std::is_integral_v<T>>>
  constexpr auto operator&&(T rhs) const -> SInt {
    return UInt<1>(static_cast<bool>(value_) && static_cast<bool>(rhs));
  }

  template <typename T, typename = std::enable_if_t<std::is_integral_v<T>>>
  constexpr auto operator||(T rhs) const -> SInt {
    return UInt<1>(static_cast<bool>(value_) || static_cast<bool>(rhs));
  }

  template <std::size_t OtherWidth>
  constexpr auto operator+(SInt<OtherWidth> rhs) const -> SInt {
    return SInt(value_ + rhs.raw());
  }

  template <std::size_t OtherWidth>
  constexpr auto operator-(SInt<OtherWidth> rhs) const -> SInt {
    return SInt(value_ - rhs.raw());
  }

  template <std::size_t OtherWidth>
  constexpr auto operator*(SInt<OtherWidth> rhs) const -> SInt {
    return SInt(value_ * rhs.raw());
  }

  template <std::size_t OtherWidth>
  constexpr auto operator/(SInt<OtherWidth> rhs) const -> SInt {
    return rhs.raw() == 0 ? SInt(0) : SInt(value_ / rhs.raw());
  }

  template <std::size_t OtherWidth>
  constexpr auto operator%(SInt<OtherWidth> rhs) const -> SInt {
    return rhs.raw() == 0 ? SInt(0) : SInt(value_ % rhs.raw());
  }

  template <std::size_t OtherWidth>
  constexpr auto operator&(SInt<OtherWidth> rhs) const -> SInt {
    return SInt(value_ & rhs.raw());
  }

  template <std::size_t OtherWidth>
  constexpr auto operator|(SInt<OtherWidth> rhs) const -> SInt {
    return SInt(value_ | rhs.raw());
  }

  template <std::size_t OtherWidth>
  constexpr auto operator^(SInt<OtherWidth> rhs) const -> SInt {
    return SInt(value_ ^ rhs.raw());
  }

  template <std::size_t OtherWidth>
  constexpr auto operator<<(SInt<OtherWidth> rhs) const -> SInt {
    return SInt(value_ << rhs.raw());
  }

  template <std::size_t OtherWidth>
  constexpr auto operator>>(SInt<OtherWidth> rhs) const -> SInt {
    return SInt(value_ >> rhs.raw());
  }

  template <std::size_t OtherWidth>
  constexpr auto operator+=(SInt<OtherWidth> rhs) -> SInt & {
    value_ = normalize(value_ + rhs.raw());
    return *this;
  }

  template <std::size_t OtherWidth>
  constexpr auto operator-=(SInt<OtherWidth> rhs) -> SInt & {
    value_ = normalize(value_ - rhs.raw());
    return *this;
  }

  template <std::size_t OtherWidth>
  constexpr auto operator*=(SInt<OtherWidth> rhs) -> SInt & {
    value_ = normalize(value_ * rhs.raw());
    return *this;
  }

  template <std::size_t OtherWidth>
  constexpr auto operator/=(SInt<OtherWidth> rhs) -> SInt & {
    value_ = rhs.raw() == 0 ? 0 : normalize(value_ / rhs.raw());
    return *this;
  }

  template <std::size_t OtherWidth>
  constexpr auto operator%=(SInt<OtherWidth> rhs) -> SInt & {
    value_ = rhs.raw() == 0 ? 0 : normalize(value_ % rhs.raw());
    return *this;
  }

  template <std::size_t OtherWidth>
  constexpr auto operator&=(SInt<OtherWidth> rhs) -> SInt & {
    value_ = normalize(value_ & rhs.raw());
    return *this;
  }

  template <std::size_t OtherWidth>
  constexpr auto operator|=(SInt<OtherWidth> rhs) -> SInt & {
    value_ = normalize(value_ | rhs.raw());
    return *this;
  }

  template <std::size_t OtherWidth>
  constexpr auto operator^=(SInt<OtherWidth> rhs) -> SInt & {
    value_ = normalize(value_ ^ rhs.raw());
    return *this;
  }

  template <std::size_t OtherWidth>
  constexpr auto operator<<=(SInt<OtherWidth> rhs) -> SInt & {
    value_ = normalize(value_ << rhs.raw());
    return *this;
  }

  template <std::size_t OtherWidth>
  constexpr auto operator>>=(SInt<OtherWidth> rhs) -> SInt & {
    value_ = normalize(value_ >> rhs.raw());
    return *this;
  }

  template <std::size_t OtherWidth>
  constexpr auto operator==(SInt<OtherWidth> rhs) const -> bool {
    return value_ == rhs.raw();
  }

  template <std::size_t OtherWidth>
  constexpr auto operator!=(SInt<OtherWidth> rhs) const -> bool {
    return value_ != rhs.raw();
  }

  template <std::size_t OtherWidth>
  constexpr auto operator<(SInt<OtherWidth> rhs) const -> bool {
    return value_ < rhs.raw();
  }

  template <std::size_t OtherWidth>
  constexpr auto operator<=(SInt<OtherWidth> rhs) const -> bool {
    return value_ <= rhs.raw();
  }

  template <std::size_t OtherWidth>
  constexpr auto operator>(SInt<OtherWidth> rhs) const -> bool {
    return value_ > rhs.raw();
  }

  template <std::size_t OtherWidth>
  constexpr auto operator>=(SInt<OtherWidth> rhs) const -> bool {
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

  constexpr auto operator++() -> SInt & {
    *this += SInt(1);
    return *this;
  }

  constexpr auto operator++(int) -> SInt {
    SInt old = *this;
    ++(*this);
    return old;
  }

  constexpr auto operator--() -> SInt & {
    *this -= SInt(1);
    return *this;
  }

  constexpr auto operator--(int) -> SInt {
    SInt old = *this;
    --(*this);
    return old;
  }

  static constexpr auto mask() -> storage_type {
    if constexpr (Width == 64) {
      return ~storage_type{0};
    } else {
      return (storage_type{1} << Width) - 1;
    }
  }

  static constexpr auto normalize(storage_type value) -> storage_type {
    if constexpr (Width == 64) {
      return value;
    } else {
      constexpr int shift = 64 - Width;
      return (value << shift) >> shift;
    }
  }

private:
  storage_type value_{0};
};

template <std::size_t Width> struct TypeTraits<SInt<Width>> {
  static constexpr bool is_chwc_type = true;
  static constexpr bool is_signal = false;
  static constexpr bool is_signed = true;
  static constexpr std::size_t width = Width;
  static constexpr ObjectKind kind = ObjectKind::Value;

  using value_type = SInt<Width>;
};

} // namespace chwc

#endif // CHWC_RUNTIME_SINT_H
