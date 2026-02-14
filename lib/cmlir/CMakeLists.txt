add_subdirectory(Transforms)

add_mlir_dialect_library(cmlir 
  Transforms/Mem2Reg.cc
  Transforms/Raise2Affine.cc
  Dialect.cc

  DEPENDS 
  MLIRCMLIROpsIncGen
  MLIRCMLIRTransformsIncGen
)
