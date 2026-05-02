#ifndef CHWC_RUNTIME_RUNTIME_H
#define CHWC_RUNTIME_RUNTIME_H

#include "chwc/Annotation.h"
#include "chwc/Hardware.h"
#include "chwc/Type.h"

#ifndef CHWC_NO_GLOBAL_HARDWARE_ALIAS
using Hardware = ::chwc::Hardware;
#endif

#endif // CHWC_RUNTIME_RUNTIME_H
