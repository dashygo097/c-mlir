#ifndef CHWC_RUNTIME_RUNTIME_H
#define CHWC_RUNTIME_RUNTIME_H

#include "chwc/Annotation.h"
#include "chwc/Module.h"
#include "chwc/Ops/Delay.h"
#include "chwc/Ops/Mux.h"
#include "chwc/Ops/WireDefault.h"
#include "chwc/Signal.h"
#include "chwc/Types/Enum.h"
#include "chwc/Types/SInt.h"
#include "chwc/Types/UInt.h"

#ifndef CHWC_NO_GLOBAL_MODULE_ALIAS
using ::chwc::Instance;
using ::chwc::Module;
#endif

#ifndef CHWC_NO_GLOBAL_TYPE_ALIAS
using ::chwc::Input;
using ::chwc::Output;
using ::chwc::Reg;
using ::chwc::Wire;

using ::chwc::Bool;
using ::chwc::Enum;
using ::chwc::SInt;
using ::chwc::UInt;

#endif

#ifndef CHWC_NO_GLOBAL_OPS_ALIAS
using ::chwc::Delay;
using ::chwc::Mux;
using ::chwc::RegNext;
using ::chwc::WireDefault;
#endif

#endif // CHWC_RUNTIME_RUNTIME_H
