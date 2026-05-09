#ifndef CHWC_RUNTIME_TYPES_ENUM_H
#define CHWC_RUNTIME_TYPES_ENUM_H

#include "chwc/Types/UInt.h"

namespace chwc {

namespace impl {
constexpr auto enumBits(std::size_t number) -> std::size_t {
  std::size_t bits = 0;
  std::size_t values = 1;

  while (values < number) {
    values <<= 1;
    ++bits;
  }

  return bits == 0 ? 1 : bits;
}
} // namespace impl

template <std::size_t Number> using Enum = UInt<impl::enumBits(Number)>;
using Bool = UInt<1>;

} // namespace chwc

#endif // CHWC_RUNTIME_TYPES_ENUM_H
