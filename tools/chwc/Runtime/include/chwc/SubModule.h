#ifndef CHWC_RUNTIME_SUBMODULE_H
#define CHWC_RUNTIME_SUBMODULE_H

#include "chwc/Module.h"
#include <type_traits>

namespace chwc {

template <typename M> class SubModule : M {
public:
  static_assert(std::is_base_of_v<Module, M>,
                "Submodule must derive from Module");

  M io;
};

} // namespace chwc

#endif // CHWC_RUNTIME_SUBMODULE_H
