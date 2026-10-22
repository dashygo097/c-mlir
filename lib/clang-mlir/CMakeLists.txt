add_library(ClangMLIRSupport STATIC Source.cc Address.cc Regions.cc Casts.cc MemrefABI.cc FloatingPoint.cc)
target_include_directories(ClangMLIRSupport PUBLIC ${PROJECT_SOURCE_DIR}/include)
target_link_libraries(ClangMLIRSupport PUBLIC clangAST clangBasic MLIRIR MLIRArithDialect MLIRMemRefDialect MLIRLLVMDialect MLIRControlFlowDialect MLIRSCFDialect)

llvm_update_compile_flags(ClangMLIRSupport)

add_library(CMLIR::ClangMLIRSupport ALIAS ClangMLIRSupport)

target_include_directories(ClangMLIRSupport SYSTEM PUBLIC ${LLVM_INCLUDE_DIRS} ${MLIR_INCLUDE_DIRS} ${CLANG_INCLUDE_DIRS})

target_compile_features(ClangMLIRSupport PUBLIC cxx_std_17)
