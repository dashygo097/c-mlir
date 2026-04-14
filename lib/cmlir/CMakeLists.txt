add_mlir_dialect_library(cmlir 
  Transforms/ConstProp.cc

  Transforms/LoopUnroll.cc
  Transforms/LoopVectorize.cc

  Transforms/ArithCastProp.cc
  Transforms/Struct2Memref.cc
  Transforms/FMA.cc
  Transforms/RaiseMemref2Affine.cc
  Transforms/RaiseSCF2Affine.cc
  Dialect.cc

  DEPENDS 
  MLIRCMLIROpsIncGen
  MLIRCMLIRTransformsIncGen
)
