add_mlir_dialect_library(cmlir 
  Transforms/Mem2Reg.cc
  Transforms/ConstProp.cc
  Transforms/FMA.cc
  Dialect.cc

  DEPENDS 
  MLIRCMLIROpsIncGen
  MLIRCMLIRTransformsIncGen
)
