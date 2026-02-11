set(CMLIR_TEST_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR})
set(CMLIR_TEST_BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR})

configure_lit_site_cfg(
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.py.in
  ${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg.py
  MAIN_CONFIG
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.cfg.py
)

set(CMLIR_TEST_DEPS
  FileCheck
  count
  not
  split-file
  
  # MLIR 
  mlir-opt
  mlir-translate
  
  cmlirc
  
  # Clang
  clang
)

add_lit_testsuite(check-cmlirc
  "Running the C-MLIR regression tests"
  ${CMAKE_CURRENT_BINARY_DIR}
  DEPENDS ${CMLIR_TEST_DEPS}
)

add_lit_testsuite(check-cmlirc-single
  "Running the C-MLIR regression tests (single-threaded)"
  ${CMAKE_CURRENT_BINARY_DIR}
  DEPENDS ${CMLIR_TEST_DEPS}
  ARGS -j 1 -v
)

set_target_properties(check-cmlirc PROPERTIES FOLDER "Tests")
set_target_properties(check-cmlirc-single PROPERTIES FOLDER "Tests")
