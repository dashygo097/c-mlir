add_mlir_dialect_library(cmlir 
  Transforms/Mem2Reg.cc
  Transforms/ConstProp.cc
  Transforms/FMA.cc
  Transforms/Struct2Memref.cc
  Transforms/LoopUnroll.cc
  Transforms/RaiseSCF2Affine.cc
  Dialect.cc

  DEPENDS 
  MLIRCMLIROpsIncGen
  MLIRCMLIRTransformsIncGen
)
