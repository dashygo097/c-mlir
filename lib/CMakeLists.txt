set(CMLIR_SOURCES
  cmlir.cc
)

add_library(cmlir ${CMLIR_SOURCES})

target_include_directories(cmlir
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

