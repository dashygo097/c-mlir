add_mlir_library(CMLIRTransforms
  ArithCastProp.cc
  ConstProp.cc
  FMA.cc
  LoopUnroll.cc
  LoopVectorize.cc
  Pipeline.cc
  RaiseMemref2Affine.cc
  RaiseSCF2Affine.cc
  Struct2Memref.cc
  DEPENDS MLIRCMLIRTransformsIncGen
  LINK_LIBS PUBLIC
    MLIRSupport
    MLIRIR
    MLIRPass
    MLIRTransforms
    MLIRTransformUtils
    MLIRAffineDialect
    MLIRAffineTransforms
    MLIRArithDialect
    MLIRArithTransforms
    MLIRBufferizationTransforms
    MLIRFuncDialect
    MLIRFuncInlinerExtension
    MLIRLLVMDialect
    MLIRLLVMIRTransforms
    MLIRMathDialect
    MLIRMemRefDialect
    MLIRMemRefTransforms
    MLIRSCFDialect
    MLIRSCFTransforms
    MLIRVectorDialect
)
target_include_directories(CMLIRTransforms PUBLIC
  ${PROJECT_SOURCE_DIR}/include
  ${PROJECT_BINARY_DIR}/include
)
target_compile_features(CMLIRTransforms PUBLIC cxx_std_17)
add_library(CMLIR::Transforms ALIAS CMLIRTransforms)
