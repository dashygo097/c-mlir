#ifndef CMLIR_DIALECT
#define CMLIR_DIALECT

include "mlir/IR/OpBase.td"

def CMLIRDialect : Dialect {
  let name = "cmlir";
  let description = [{}];
  let cppNamespace = "::cmlir";
}

#endif // CMLIR_DIALECT
