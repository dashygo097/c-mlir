add_mlir_library(CMLIRTransforms
  ArithCastProp.cc
  IndexCastProp.cc
  LoopUnroll.cc
  LoopVectorize.cc
  Pipeline.cc
  Struct2Memref.cc
  DEPENDS MLIRCMLIRTransformsIncGen
  LINK_LIBS PUBLIC
    CMLIRSupport
    MLIRSupport
    MLIRAnalysis
    MLIRDataLayoutInterfaces
    MLIRIR
    MLIRPass
    MLIRTransforms
    MLIRTransformUtils
    MLIRAffineDialect
    MLIRAffineTransforms
    MLIRAffineToStandard
    MLIRArithDialect
    MLIRArithTransforms
    MLIRBufferizationTransforms
    MLIRControlFlowToSCF
    MLIRFuncDialect
    MLIRFuncInlinerExtension
    MLIRLLVMDialect
    MLIRLLVMIRTransforms
    MLIRMathDialect
    MLIRMathTransforms
    MLIRMemRefDialect
    MLIRMemRefTransforms
    MLIRSCFDialect
    MLIRSCFTransforms
    MLIRSCFToAffine
    MLIRRewrite
    MLIRVectorDialect
)
target_include_directories(CMLIRTransforms PUBLIC
  ${PROJECT_SOURCE_DIR}/include
  ${PROJECT_BINARY_DIR}/include
)
target_compile_features(CMLIRTransforms PUBLIC cxx_std_17)
add_library(CMLIR::Transforms ALIAS CMLIRTransforms)
