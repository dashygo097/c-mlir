if(ENABLE_EXECUTABLES)
  configure_lit_site_cfg(
    ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.py.in
    ${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg.py
    MAIN_CONFIG ${CMAKE_CURRENT_SOURCE_DIR}/lit.cfg.py)

  set(CMLIR_TEST_DEPS
    FileCheck count not split-file mlir-opt mlir-translate lli clang clang-to-mlir)
  add_lit_testsuite(check-Frontend
    "Running frontend regression tests"
    ${CMAKE_CURRENT_BINARY_DIR}/Frontend
    DEPENDS ${CMLIR_TEST_DEPS})
  add_lit_testsuite(check-Frontend-single
    "Running frontend regression tests (single-threaded)"
    ${CMAKE_CURRENT_BINARY_DIR}/Frontend
    DEPENDS ${CMLIR_TEST_DEPS}
    ARGS -j 1 -v)
  set_target_properties(check-Frontend check-Frontend-single PROPERTIES FOLDER "Tests")
endif()
