add_library(ClangASTSupport STATIC Discovery.cc)
llvm_update_compile_flags(ClangASTSupport)
target_include_directories(ClangASTSupport PUBLIC ${PROJECT_SOURCE_DIR}/include)
target_link_libraries(ClangASTSupport PUBLIC clangAST clangBasic)

add_library(CMLIR::ClangASTSupport ALIAS ClangASTSupport)

target_include_directories(ClangASTSupport SYSTEM PUBLIC ${LLVM_INCLUDE_DIRS} ${CLANG_INCLUDE_DIRS})

target_compile_features(ClangASTSupport PUBLIC cxx_std_17)
