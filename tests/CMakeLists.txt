if(ENABLE_EXECUTABLES)
  configure_lit_site_cfg(
    ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.py.in
    ${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg.py
    MAIN_CONFIG ${CMAKE_CURRENT_SOURCE_DIR}/lit.cfg.py)

  set(CMLIR_TEST_TOOLS FileCheck count not split-file)
  set(CMLIR_TEST_DEPS ${CMLIR_TEST_TOOLS}
    mlir-opt mlir-translate lli clang clang-to-mlir cmlir-opt)
  add_lit_testsuite(check-cmlir
    "Running all CMLIR regression tests"
    ${CMAKE_CURRENT_BINARY_DIR}
    DEPENDS ${CMLIR_TEST_DEPS})
  foreach(suite Frontend Transforms Pipelines Runtime)
    if(suite STREQUAL "Transforms")
      set(suite_deps ${CMLIR_TEST_TOOLS} cmlir-opt)
    elseif(suite STREQUAL "Frontend")
      set(suite_deps ${CMLIR_TEST_TOOLS} clang-to-mlir)
    elseif(suite STREQUAL "Pipelines")
      set(suite_deps ${CMLIR_TEST_TOOLS} clang-to-mlir cmlir-opt mlir-opt)
    else()
      set(suite_deps ${CMLIR_TEST_DEPS})
    endif()
    add_lit_testsuite(check-${suite}
      "Running ${suite} regression tests"
      ${CMAKE_CURRENT_BINARY_DIR}/${suite}
      DEPENDS ${suite_deps})
    add_lit_testsuite(check-${suite}-single
      "Running ${suite} regression tests (single-threaded)"
      ${CMAKE_CURRENT_BINARY_DIR}/${suite}
      DEPENDS ${suite_deps}
      ARGS -j 1 -v)
    set_target_properties(check-${suite} check-${suite}-single
      PROPERTIES FOLDER "Tests")
  endforeach()
  set_target_properties(check-cmlir PROPERTIES FOLDER "Tests")
endif()
