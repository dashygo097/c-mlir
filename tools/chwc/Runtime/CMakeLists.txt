add_library(chwc-runtime INTERFACE)

target_include_directories(chwc-runtime
  INTERFACE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_compile_features(chwc-runtime
  INTERFACE
    cxx_std_17
)
