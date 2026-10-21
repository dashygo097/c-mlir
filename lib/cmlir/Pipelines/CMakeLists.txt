add_mlir_library(CMLIRPipelines
  DefaultPipeline.cc
  Registration.cc
  LINK_LIBS PUBLIC
    CMLIRTransforms
    CMLIRSupport
    MLIRBufferizationTransforms
    MLIRControlFlowToSCF
    MLIRFuncInlinerExtension
    MLIRLLVMIRTransforms
    MLIRMathTransforms
    MLIRSCFTransforms
    MLIRSCFToAffine
)
target_include_directories(CMLIRPipelines PUBLIC
  ${PROJECT_SOURCE_DIR}/include
  ${PROJECT_BINARY_DIR}/include
)
target_compile_features(CMLIRPipelines PUBLIC cxx_std_17)
add_library(CMLIR::Pipelines ALIAS CMLIRPipelines)
