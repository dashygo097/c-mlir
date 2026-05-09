#ifndef CHWC_RUNTIME_RUNTIME_H
#define CHWC_RUNTIME_RUNTIME_H

#include "chwc/Annotation.h"
#include "chwc/Instance.h"
#include "chwc/Module.h"
#include "chwc/Signal.h"
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

using ::chwc::SInt;
using ::chwc::UInt;

using Bool = UInt<1>;
#endif

#endif // CHWC_RUNTIME_RUNTIME_H
