add_subdirectory(Transforms)

add_mlir_dialect_library(cmlir 
  Transforms/ConstMerge.cc
  Dialect.cc

  DEPENDS 
  MLIRCMLIROpsIncGen
  MLIRCMLIRTransformsIncGen
)
