#ifndef CHWC_RUNTIME_TYPE_H
#define CHWC_RUNTIME_TYPE_H

#include <cstddef>
#include <cstdint>
#include <limits>
#include <type_traits>
#include <utility>

namespace chwc {

enum class ObjectKind {
  Value,
  Input,
  Output,
  Wire,
  Reg,
};

template <std::size_t Width> class UInt {
public:
  static_assert(Width >= 1, "UInt width must be positive");
  static_assert(Width <= 64, "UInt currently supports width <= 64");

  using storage_type = std::uint64_t;

  static constexpr std::size_t width = Width;
  static constexpr bool is_signed = false;
  static constexpr ObjectKind kind = ObjectKind::Value;

  constexpr UInt() = default;

  template <typename T, typename = std::enable_if_t<std::is_integral<T>::value>>
  constexpr UInt(T value)
      : value_(normalize(static_cast<storage_type>(value))) {}

  constexpr auto raw() const -> storage_type { return value_; }

  constexpr auto value() const -> storage_type { return value_; }

  constexpr explicit operator bool() const { return value_ != 0; }

  constexpr explicit operator storage_type() const { return value_; }

  template <typename T, typename = std::enable_if_t<std::is_integral<T>::value>>
  constexpr auto operator=(T value) -> UInt & {
    value_ = normalize(static_cast<storage_type>(value));
    return *this;
  }

  constexpr auto operator+() const -> UInt { return *this; }

  constexpr auto operator-() const -> UInt { return UInt(0 - value_); }

  constexpr auto operator~() const -> UInt { return UInt(~value_); }

  constexpr auto operator+(UInt rhs) const -> UInt {
    return UInt(value_ + rhs.value_);
  }

  constexpr auto operator-(UInt rhs) const -> UInt {
    return UInt(value_ - rhs.value_);
  }

  constexpr auto operator*(UInt rhs) const -> UInt {
    return UInt(value_ * rhs.value_);
  }

  constexpr auto operator/(UInt rhs) const -> UInt {
    return rhs.value_ == 0 ? UInt(0) : UInt(value_ / rhs.value_);
  }

  constexpr auto operator%(UInt rhs) const -> UInt {
    return rhs.value_ == 0 ? UInt(0) : UInt(value_ % rhs.value_);
  }

  constexpr auto operator&(UInt rhs) const -> UInt {
    return UInt(value_ & rhs.value_);
  }

  constexpr auto operator|(UInt rhs) const -> UInt {
    return UInt(value_ | rhs.value_);
  }

  constexpr auto operator^(UInt rhs) const -> UInt {
    return UInt(value_ ^ rhs.value_);
  }

  constexpr auto operator<<(UInt rhs) const -> UInt {
    return UInt(value_ << rhs.value_);
  }

  constexpr auto operator>>(UInt rhs) const -> UInt {
    return UInt(value_ >> rhs.value_);
  }

  constexpr auto operator+=(UInt rhs) -> UInt & {
    value_ = normalize(value_ + rhs.value_);
    return *this;
  }

  constexpr auto operator-=(UInt rhs) -> UInt & {
    value_ = normalize(value_ - rhs.value_);
    return *this;
  }

  constexpr auto operator*=(UInt rhs) -> UInt & {
    value_ = normalize(value_ * rhs.value_);
    return *this;
  }

  constexpr auto operator/=(UInt rhs) -> UInt & {
    value_ = rhs.value_ == 0 ? 0 : normalize(value_ / rhs.value_);
    return *this;
  }

  constexpr auto operator%=(UInt rhs) -> UInt & {
    value_ = rhs.value_ == 0 ? 0 : normalize(value_ % rhs.value_);
    return *this;
  }

  constexpr auto operator&=(UInt rhs) -> UInt & {
    value_ = normalize(value_ & rhs.value_);
    return *this;
  }

  constexpr auto operator|=(UInt rhs) -> UInt & {
    value_ = normalize(value_ | rhs.value_);
    return *this;
  }

  constexpr auto operator^=(UInt rhs) -> UInt & {
    value_ = normalize(value_ ^ rhs.value_);
    return *this;
  }

  constexpr auto operator<<=(UInt rhs) -> UInt & {
    value_ = normalize(value_ << rhs.value_);
    return *this;
  }

  constexpr auto operator>>=(UInt rhs) -> UInt & {
    value_ = normalize(value_ >> rhs.value_);
    return *this;
  }

  constexpr auto operator==(UInt rhs) const -> bool {
    return value_ == rhs.value_;
  }

  constexpr auto operator!=(UInt rhs) const -> bool {
    return value_ != rhs.value_;
  }

  constexpr auto operator<(UInt rhs) const -> bool {
    return value_ < rhs.value_;
  }

  constexpr auto operator<=(UInt rhs) const -> bool {
    return value_ <= rhs.value_;
  }

  constexpr auto operator>(UInt rhs) const -> bool {
    return value_ > rhs.value_;
  }

  constexpr auto operator>=(UInt rhs) const -> bool {
    return value_ >= rhs.value_;
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

template <typename T, ObjectKind Kind> class Signal {
public:
  using value_type = T;
  using storage_type = typename T::storage_type;

  static constexpr std::size_t width = T::width;
  static constexpr bool is_signed = false;
  static constexpr ObjectKind kind = Kind;

  constexpr Signal() = default;

  constexpr Signal(const T &value) : value_(value) {}

  template <typename U, typename = std::enable_if_t<!std::is_same<
                            Signal<T, Kind>, std::decay_t<U>>::value>>
  constexpr Signal(U &&value) : value_(T(std::forward<U>(value))) {}

  constexpr auto read() const -> T { return value_; }

  constexpr auto raw() const -> storage_type { return value_.raw(); }

  constexpr auto value() const -> T { return value_; }

  constexpr explicit operator bool() const { return static_cast<bool>(value_); }

  constexpr operator T() const { return value_; }

  constexpr auto operator=(const T &value) -> Signal & {
    value_ = value;
    return *this;
  }

  template <typename U, typename = std::enable_if_t<!std::is_same<
                            Signal<T, Kind>, std::decay_t<U>>::value>>
  constexpr auto operator=(U &&value) -> Signal & {
    value_ = T(std::forward<U>(value));
    return *this;
  }

  constexpr auto operator+() const -> T { return value_; }

  constexpr auto operator-() const -> T { return -value_; }

  constexpr auto operator~() const -> T { return ~value_; }

  constexpr auto operator!() const -> bool {
    return !static_cast<bool>(value_);
  }

  constexpr auto operator++() -> Signal & {
    ++value_;
    return *this;
  }

  constexpr auto operator++(int) -> T {
    T old = value_;
    ++value_;
    return old;
  }

  constexpr auto operator--() -> Signal & {
    --value_;
    return *this;
  }

  constexpr auto operator--(int) -> T {
    T old = value_;
    --value_;
    return old;
  }

  template <typename U> constexpr auto operator+=(const U &rhs) -> Signal & {
    value_ = value_ + unwrap(rhs);
    return *this;
  }

  template <typename U> constexpr auto operator-=(const U &rhs) -> Signal & {
    value_ = value_ - unwrap(rhs);
    return *this;
  }

  template <typename U> constexpr auto operator*=(const U &rhs) -> Signal & {
    value_ = value_ * unwrap(rhs);
    return *this;
  }

  template <typename U> constexpr auto operator/=(const U &rhs) -> Signal & {
    value_ = value_ / unwrap(rhs);
    return *this;
  }

  template <typename U> constexpr auto operator%=(const U &rhs) -> Signal & {
    value_ = value_ % unwrap(rhs);
    return *this;
  }

  template <typename U> constexpr auto operator&=(const U &rhs) -> Signal & {
    value_ = value_ & unwrap(rhs);
    return *this;
  }

  template <typename U> constexpr auto operator|=(const U &rhs) -> Signal & {
    value_ = value_ | unwrap(rhs);
    return *this;
  }

  template <typename U> constexpr auto operator^=(const U &rhs) -> Signal & {
    value_ = value_ ^ unwrap(rhs);
    return *this;
  }

  template <typename U> constexpr auto operator<<=(const U &rhs) -> Signal & {
    value_ = value_ << unwrap(rhs);
    return *this;
  }

  template <typename U> constexpr auto operator>>=(const U &rhs) -> Signal & {
    value_ = value_ >> unwrap(rhs);
    return *this;
  }

  template <typename U> constexpr auto operator+(const U &rhs) const {
    return value_ + unwrap(rhs);
  }

  template <typename U> constexpr auto operator-(const U &rhs) const {
    return value_ - unwrap(rhs);
  }

  template <typename U> constexpr auto operator*(const U &rhs) const {
    return value_ * unwrap(rhs);
  }

  template <typename U> constexpr auto operator/(const U &rhs) const {
    return value_ / unwrap(rhs);
  }

  template <typename U> constexpr auto operator%(const U &rhs) const {
    return value_ % unwrap(rhs);
  }

  template <typename U> constexpr auto operator&(const U &rhs) const {
    return value_ & unwrap(rhs);
  }

  template <typename U> constexpr auto operator|(const U &rhs) const {
    return value_ | unwrap(rhs);
  }

  template <typename U> constexpr auto operator^(const U &rhs) const {
    return value_ ^ unwrap(rhs);
  }

  template <typename U> constexpr auto operator<<(const U &rhs) const {
    return value_ << unwrap(rhs);
  }

  template <typename U> constexpr auto operator>>(const U &rhs) const {
    return value_ >> unwrap(rhs);
  }

  template <typename U> constexpr auto operator==(const U &rhs) const -> bool {
    return value_ == unwrap(rhs);
  }

  template <typename U> constexpr auto operator!=(const U &rhs) const -> bool {
    return value_ != unwrap(rhs);
  }

  template <typename U> constexpr auto operator<(const U &rhs) const -> bool {
    return value_ < unwrap(rhs);
  }

  template <typename U> constexpr auto operator<=(const U &rhs) const -> bool {
    return value_ <= unwrap(rhs);
  }

  template <typename U> constexpr auto operator>(const U &rhs) const -> bool {
    return value_ > unwrap(rhs);
  }

  template <typename U> constexpr auto operator>=(const U &rhs) const -> bool {
    return value_ >= unwrap(rhs);
  }

private:
  template <typename U> static constexpr auto unwrap(const U &value) -> T {
    return T(value);
  }

  template <ObjectKind OtherKind>
  static constexpr auto unwrap(const Signal<T, OtherKind> &signal) -> T {
    return signal.read();
  }

  T value_{};
};

template <typename T> using Input = Signal<T, ObjectKind::Input>;

template <typename T> using Output = Signal<T, ObjectKind::Output>;

template <typename T> using Wire = Signal<T, ObjectKind::Wire>;

template <typename T> using Reg = Signal<T, ObjectKind::Reg>;

template <typename T> struct TypeTraits {
  static constexpr bool is_chwc_type = false;
};

template <std::size_t Width> struct TypeTraits<UInt<Width>> {
  static constexpr bool is_chwc_type = true;
  static constexpr bool is_signal = false;
  static constexpr bool is_signed = false;
  static constexpr std::size_t width = Width;
  static constexpr ObjectKind kind = ObjectKind::Value;

  using value_type = UInt<Width>;
};

template <typename T, ObjectKind Kind> struct TypeTraits<Signal<T, Kind>> {
  static constexpr bool is_chwc_type = true;
  static constexpr bool is_signal = true;
  static constexpr bool is_signed = false;
  static constexpr std::size_t width = T::width;
  static constexpr ObjectKind kind = Kind;

  using value_type = T;
};

} // namespace chwc

#endif // CHWC_RUNTIME_TYPE_H
