add_mlir_dialect_library(CMLIRDialect
  Dialect.cc
  DEPENDS MLIRCMLIROpsIncGen
  LINK_LIBS PUBLIC MLIRIR MLIRSideEffectInterfaces
)
target_include_directories(CMLIRDialect PUBLIC
  ${PROJECT_SOURCE_DIR}/include
  ${PROJECT_BINARY_DIR}/include
)
target_compile_features(CMLIRDialect PUBLIC cxx_std_17)
add_library(CMLIR::Dialect ALIAS CMLIRDialect)

add_subdirectory(Transforms)
if(BUILD_FRONTEND)
  add_subdirectory(Frontend)
endif()
