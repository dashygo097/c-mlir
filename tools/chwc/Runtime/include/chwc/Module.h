#ifndef CHWC_RUNTIME_MODULE_H
#define CHWC_RUNTIME_MODULE_H

#include <cstddef>
#include <type_traits>

namespace chwc {

class Module {};

template <typename M> class Instance : M {
public:
  static_assert(std::is_base_of_v<Module, M>,
                "Submodule must derive from Module");

  M io;
};

} // namespace chwc

#endif // CHWC_RUNTIME_MODULE_H
