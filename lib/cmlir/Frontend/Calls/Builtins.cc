#include "../Lowering.h"
#include "CallLowering.h"
#include "clang-mlir/Casts.h"
#include "clang-mlir/Constants.h"
#include "clang-mlir/MemrefABI.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/WithColor.h"

#include "mlir/IR/Diagnostics.h"

namespace cmlir::frontend {

auto tryEmitStdlibCall(mlir::OpBuilder &builder, mlir::Location loc,
                       mlir::ModuleOp module, llvm::StringRef name,
                       llvm::ArrayRef<CallArg> args, mlir::Value &outResult)
    -> bool {
  mlir::MLIRContext *ctx = builder.getContext();

  auto i32 = builder.getI32Type();
  auto i64 = builder.getI64Type();
  auto f64 = builder.getF64Type();
  auto addr = mlir::LLVM::LLVMPointerType::get(ctx);
  auto voidTy = mlir::LLVM::LLVMVoidType::get(ctx);

  auto emit = [&](llvm::ArrayRef<mlir::Type> fixedArgTypes, mlir::Type retType,
                  bool isVarArg) -> bool {
    return emitExternalCall(builder, loc, module, name, args, fixedArgTypes,
                            retType, isVarArg, outResult);
  };

#define VOID0() return emit(llvm::ArrayRef<mlir::Type>{}, voidTy, false)
#define RET0(r) return emit(llvm::ArrayRef<mlir::Type>{}, r, false)
#define VOID(...) return emit({__VA_ARGS__}, voidTy, false)
#define RET(r, ...) return emit({__VA_ARGS__}, r, false)
#define VRET(r, ...) return emit({__VA_ARGS__}, r, true)

  if (name == "printf") {
    VRET(i32, addr);
  }
  if (name == "fprintf") {
    VRET(i32, addr, addr);
  }
  if (name == "sprintf") {
    VRET(i32, addr, addr);
  }
  if (name == "snprintf") {
    VRET(i32, addr, i64, addr);
  }
  if (name == "scanf") {
    VRET(i32, addr);
  }
  if (name == "fscanf") {
    VRET(i32, addr, addr);
  }
  if (name == "sscanf") {
    VRET(i32, addr, addr);
  }
  if (name == "puts") {
    RET(i32, addr);
  }
  if (name == "putchar") {
    RET(i32, i32);
  }
  if (name == "getchar") {
    RET0(i32);
  }
  if (name == "gets") {
    RET(addr, addr);
  }
  if (name == "fgets") {
    RET(addr, addr, i32, addr);
  }
  if (name == "fputs") {
    RET(i32, addr, addr);
  }
  if (name == "fopen") {
    RET(addr, addr, addr);
  }
  if (name == "fclose") {
    RET(i32, addr);
  }
  if (name == "fread") {
    RET(i64, addr, i64, i64, addr);
  }
  if (name == "fwrite") {
    RET(i64, addr, i64, i64, addr);
  }
  if (name == "fseek") {
    RET(i32, addr, i64, i32);
  }
  if (name == "ftell") {
    RET(i64, addr);
  }
  if (name == "rewind") {
    VOID(addr);
  }
  if (name == "feof") {
    RET(i32, addr);
  }
  if (name == "ferror") {
    RET(i32, addr);
  }
  if (name == "fflush") {
    RET(i32, addr);
  }
  if (name == "remove") {
    RET(i32, addr);
  }
  if (name == "rename") {
    RET(i32, addr, addr);
  }
  if (name == "perror") {
    VOID(addr);
  }

  if (name == "malloc") {
    RET(addr, i64);
  }
  if (name == "calloc") {
    RET(addr, i64, i64);
  }
  if (name == "realloc") {
    RET(addr, addr, i64);
  }
  if (name == "free") {
    VOID(addr);
  }
  if (name == "exit") {
    VOID(i32);
  }
  if (name == "abort") {
    VOID0();
  }
  if (name == "rand") {
    RET0(i32);
  }
  if (name == "srand") {
    VOID(i32);
  }
  if (name == "atoi") {
    RET(i32, addr);
  }
  if (name == "atol") {
    RET(i64, addr);
  }
  if (name == "atof") {
    RET(f64, addr);
  }
  if (name == "strtol") {
    RET(i64, addr, addr, i32);
  }
  if (name == "strtod") {
    RET(f64, addr, addr);
  }
  if (name == "qsort") {
    VOID(addr, i64, i64, addr);
  }
  if (name == "bsearch") {
    RET(addr, addr, addr, i64, i64, addr);
  }
  if (name == "getenv") {
    RET(addr, addr);
  }
  if (name == "system") {
    RET(i32, addr);
  }

  if (name == "strlen") {
    RET(i64, addr);
  }
  if (name == "strcpy") {
    RET(addr, addr, addr);
  }
  if (name == "strncpy") {
    RET(addr, addr, addr, i64);
  }
  if (name == "strcat") {
    RET(addr, addr, addr);
  }
  if (name == "strncat") {
    RET(addr, addr, addr, i64);
  }
  if (name == "strcmp") {
    RET(i32, addr, addr);
  }
  if (name == "strncmp") {
    RET(i32, addr, addr, i64);
  }
  if (name == "strchr") {
    RET(addr, addr, i32);
  }
  if (name == "strrchr") {
    RET(addr, addr, i32);
  }
  if (name == "strstr") {
    RET(addr, addr, addr);
  }
  if (name == "strtok") {
    RET(addr, addr, addr);
  }
  if (name == "strdup") {
    RET(addr, addr);
  }
  if (name == "memcpy") {
    RET(addr, addr, addr, i64);
  }
  if (name == "memmove") {
    RET(addr, addr, addr, i64);
  }
  if (name == "memset") {
    RET(addr, addr, i32, i64);
  }
  if (name == "memcmp") {
    RET(i32, addr, addr, i64);
  }
  if (name == "memchr") {
    RET(addr, addr, i32, i64);
  }

  if (name == "modf" || name == "modff") {
    RET(f64, f64, addr);
  }
  if (name == "frexp" || name == "frexpf") {
    RET(f64, f64, addr);
  }
  if (name == "ldexp" || name == "ldexpf") {
    RET(f64, f64, i32);
  }
  if (name == "hypot" || name == "hypotf") {
    RET(f64, f64, f64);
  }
  if (name == "fmod" || name == "fmodf") {
    RET(f64, f64, f64);
  }
  if (name == "remainder" || name == "remainderf") {
    RET(f64, f64, f64);
  }
  if (name == "fmin" || name == "fminf") {
    RET(f64, f64, f64);
  }
  if (name == "fmax" || name == "fmaxf") {
    RET(f64, f64, f64);
  }
  if (name == "fdim" || name == "fdimf") {
    RET(f64, f64, f64);
  }
  if (name == "nearbyint" || name == "nearbyintf") {
    RET(f64, f64);
  }
  if (name == "rint" || name == "rintf") {
    RET(f64, f64);
  }
  if (name == "lround" || name == "lroundf") {
    RET(i64, f64);
  }
  if (name == "lrint" || name == "lrintf") {
    RET(i64, f64);
  }
  if (name == "scalbn" || name == "scalbnf") {
    RET(f64, f64, i32);
  }
  if (name == "ilogb" || name == "ilogbf") {
    RET(i32, f64);
  }
  if (name == "logb" || name == "logbf") {
    RET(f64, f64);
  }
  if (name == "nan" || name == "nanf") {
    RET(f64, addr);
  }

  if (name == "time") {
    RET(i64, addr);
  }
  if (name == "clock") {
    RET0(i64);
  }
  if (name == "difftime") {
    RET(f64, i64, i64);
  }
  if (name == "mktime") {
    RET(i64, addr);
  }
  if (name == "gmtime") {
    RET(addr, addr);
  }
  if (name == "localtime") {
    RET(addr, addr);
  }
  if (name == "strftime") {
    RET(i64, addr, i64, addr, addr);
  }
  if (name == "asctime") {
    RET(addr, addr);
  }
  if (name == "ctime") {
    RET(addr, addr);
  }

  if (name == "read") {
    RET(i64, i32, addr, i64);
  }
  if (name == "write") {
    RET(i64, i32, addr, i64);
  }
  if (name == "close") {
    RET(i32, i32);
  }
  if (name == "open") {
    VRET(i32, addr, i32);
  }
  if (name == "sleep") {
    RET(i32, i32);
  }
  if (name == "usleep") {
    RET(i32, i32);
  }
  if (name == "getpid") {
    RET0(i32);
  }
  if (name == "getppid") {
    RET0(i32);
  }
  if (name == "fork") {
    RET0(i32);
  }
  if (name == "execv") {
    RET(i32, addr, addr);
  }
  if (name == "execvp") {
    RET(i32, addr, addr);
  }

#undef VOID0
#undef RET0
#undef VOID
#undef RET
#undef VRET

  return false;
}

} // namespace cmlir::frontend
