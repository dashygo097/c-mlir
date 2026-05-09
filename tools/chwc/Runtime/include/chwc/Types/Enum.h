#ifndef CHWC_RUNTIME_TYPES_ENUM_H
#define CHWC_RUNTIME_TYPES_ENUM_H

#include "chwc/Types/UInt.h"
#include <cmath>

namespace chwc {

using Bool = UInt<1>;

template <std::size_t Number>
using Enum = UInt<static_cast<int>(ceil(log2(Number)))>;

} // namespace chwc

#endif // CHWC_RUNTIME_TYPES_ENUM_H
