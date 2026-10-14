add_executable(CMLIRDialectTests api/DialectTest.cpp)
llvm_update_compile_flags(CMLIRDialectTests)
target_link_libraries(CMLIRDialectTests PRIVATE CMLIR::Dialect)
add_test(NAME dialect-api COMMAND CMLIRDialectTests)
add_custom_target(check-dialect-api
  COMMAND $<TARGET_FILE:CMLIRDialectTests>
  DEPENDS CMLIRDialectTests)

add_executable(CMLIRTransformsTests api/TransformsTest.cpp)
llvm_update_compile_flags(CMLIRTransformsTests)
target_link_libraries(CMLIRTransformsTests PRIVATE CMLIR::Transforms)
add_test(NAME transforms-api COMMAND CMLIRTransformsTests)
add_custom_target(check-transforms-api
  COMMAND $<TARGET_FILE:CMLIRTransformsTests>
  DEPENDS CMLIRTransformsTests)

add_custom_target(check-libraries
  DEPENDS check-dialect-api check-transforms-api)

if(BUILD_FRONTEND)
  add_executable(CMLIRFrontendTests api/FrontendTest.cpp)
  llvm_update_compile_flags(CMLIRFrontendTests)
  target_link_libraries(CMLIRFrontendTests PRIVATE CMLIR::Frontend clangTooling)
  add_test(NAME frontend-api COMMAND CMLIRFrontendTests)
  add_custom_target(check-frontend-api
    COMMAND $<TARGET_FILE:CMLIRFrontendTests>
    DEPENDS CMLIRFrontendTests
    COMMENT "Checking the reusable frontend API")
  add_dependencies(check-libraries check-frontend-api)
endif()

if(BUILD_CLANG_TO_MLIR)
  configure_lit_site_cfg(
    ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.py.in
    ${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg.py
    MAIN_CONFIG ${CMAKE_CURRENT_SOURCE_DIR}/lit.cfg.py)

  set(CMLIR_TEST_DEPS
    FileCheck count not split-file mlir-opt mlir-translate lli clang clang-to-mlir)
  add_lit_testsuite(check-frontend
    "Running frontend regression tests"
    ${CMAKE_CURRENT_BINARY_DIR}/frontend
    DEPENDS ${CMLIR_TEST_DEPS})
  add_lit_testsuite(check-frontend-single
    "Running frontend regression tests (single-threaded)"
    ${CMAKE_CURRENT_BINARY_DIR}/frontend
    DEPENDS ${CMLIR_TEST_DEPS}
    ARGS -j 1 -v)
  add_dependencies(check-frontend check-libraries)
  add_dependencies(check-frontend-single check-libraries)
  set_target_properties(check-frontend check-frontend-single PROPERTIES FOLDER "Tests")
endif()
