add_clang_executable(clang-to-mlir driver.cpp FrontendAction.cpp)
target_include_directories(clang-to-mlir PRIVATE
  ${PROJECT_SOURCE_DIR}/lib/cmlir/Frontend
)
target_link_libraries(clang-to-mlir PRIVATE
  CMLIR::Frontend
  CMLIR::Transforms
  clangTooling
)
