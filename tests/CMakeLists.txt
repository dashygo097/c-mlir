if(ENABLE_EXECUTABLES)
  configure_lit_site_cfg(
    ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.py.in
    ${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg.py
    MAIN_CONFIG ${CMAKE_CURRENT_SOURCE_DIR}/lit.cfg.py)

  set(CMLIR_TEST_DEPS
    FileCheck count not split-file mlir-opt mlir-translate lli clang clang-to-mlir)
  add_lit_testsuite(check-cmlir
    "Running all CMLIR regression tests"
    ${CMAKE_CURRENT_BINARY_DIR}
    DEPENDS ${CMLIR_TEST_DEPS})
  foreach(suite Frontend Transforms Runtime)
    add_lit_testsuite(check-${suite}
      "Running ${suite} regression tests"
      ${CMAKE_CURRENT_BINARY_DIR}/${suite}
      DEPENDS ${CMLIR_TEST_DEPS})
    add_lit_testsuite(check-${suite}-single
      "Running ${suite} regression tests (single-threaded)"
      ${CMAKE_CURRENT_BINARY_DIR}/${suite}
      DEPENDS ${CMLIR_TEST_DEPS}
      ARGS -j 1 -v)
    set_target_properties(check-${suite} check-${suite}-single
      PROPERTIES FOLDER "Tests")
  endforeach()
  set_target_properties(check-cmlir PROPERTIES FOLDER "Tests")
endif()
