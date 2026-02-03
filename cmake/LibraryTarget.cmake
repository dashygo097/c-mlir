# Build library

set(CMLIR_INCLUDE_DIR ${CMAKE_SOURCE_DIR}/include)
set(CMLIR_LIB_DIR ${CMAKE_SOURCE_DIR}/lib)

file(GLOB CMLIR_HEADERS ${CMLIR_INCLUDE_DIR}/*.hh)
file(GLOB CMLIR_SOURCES ${CMLIR_LIB_DIR}/*.cc)

include_directories(${CMLIR_INCLUDE_DIR})
include_directories(${PROJECT_BINARY_DIR}/include)

add_library(cmlir ${CMLIR_SOURCES} ${CMLIR_HEADERS})
