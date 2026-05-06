#ifndef CHWC_RUNTIME_RUNTIME_H
#define CHWC_RUNTIME_RUNTIME_H

#include "chwc/Annotation.h"
#include "chwc/Hardware.h"
#include "chwc/Types/Signal.h"
#include "chwc/Types/UInt.h"

#ifndef CHWC_NO_GLOBAL_HARDWARE_ALIAS
using Hardware = ::chwc::Hardware;
#endif

#ifndef CHWC_NO_GLOBAL_TYPE_ALIAS
using ::chwc::Input;
using ::chwc::Output;
using ::chwc::Reg;
using ::chwc::UInt;
using ::chwc::Wire;
#endif

#endif // CHWC_RUNTIME_RUNTIME_H
