#ifndef CHWC_RUNTIME_SIGNAL_H
#define CHWC_RUNTIME_SIGNAL_H

#include <cstddef>
#include <type_traits>

namespace chwc {

enum class ObjectKind {
  Value,
  Input,
  Output,
  Wire,
  Reg,
};

template <typename T, ObjectKind Kind> class Signal {
public:
  using value_type = T;
  using storage_type = typename T::storage_type;
  using index_type = typename T::index_type;

  static constexpr std::size_t width = T::width;
  static constexpr bool is_signed = T::is_signed;
  static constexpr ObjectKind kind = Kind;

  constexpr Signal() = default;

  explicit constexpr Signal(const T &value) : value_(value) {}

  template <typename U, typename = std::enable_if_t<
                            !std::is_same_v<Signal<T, Kind>, std::decay_t<U>>>>
  explicit constexpr Signal(const U &value) : value_(unwrap(value)) {}

  [[nodiscard]] constexpr auto read() const -> T { return value_; }

  [[nodiscard]] constexpr auto raw() const -> storage_type {
    return value_.raw();
  }

  [[nodiscard]] constexpr auto value() const -> T { return value_; }

  constexpr explicit operator bool() const { return static_cast<bool>(value_); }

  constexpr operator T() const { return value_; }

  constexpr operator index_type() const {
    return static_cast<index_type>(value_.raw());
  }

  constexpr auto operator=(const T &value) -> Signal & {
    value_ = value;
    return *this;
  }

  template <typename U, typename = std::enable_if_t<
                            !std::is_same_v<Signal<T, Kind>, std::decay_t<U>>>>
  constexpr auto operator=(const U &value) -> Signal & {
    value_ = unwrap(value);
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

  template <typename U> constexpr auto operator&&(const U &rhs) const -> bool {
    return static_cast<bool>(value_) && static_cast<bool>(unwrap(rhs));
  }

  template <typename U> constexpr auto operator||(const U &rhs) const -> bool {
    return static_cast<bool>(value_) || static_cast<bool>(unwrap(rhs));
  }

private:
  template <typename U> static constexpr auto unwrap(const U &value) -> T {
    return T(value);
  }

  template <typename U, ObjectKind OtherKind>
  static constexpr auto unwrap(const Signal<U, OtherKind> &signal) -> T {
    return T(signal.read());
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

template <typename T, ObjectKind Kind> struct TypeTraits<Signal<T, Kind>> {
  static constexpr bool is_chwc_type = true;
  static constexpr bool is_signal = true;
  static constexpr bool is_signed = T::is_signed;
  static constexpr std::size_t width = T::width;
  static constexpr ObjectKind kind = Kind;

  using value_type = T;
  using storage_type = typename T::storage_type;
  using index_type = typename T::index_type;
};

} // namespace chwc

#endif // CHWC_RUNTIME_SIGNAL_H
