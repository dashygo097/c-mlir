add_library(CMLIRSupport STATIC Printing.cc)
llvm_update_compile_flags(CMLIRSupport)
target_include_directories(CMLIRSupport PUBLIC
  ${PROJECT_SOURCE_DIR}/include
)
target_link_libraries(CMLIRSupport PUBLIC MLIRIR LLVMSupport)
target_compile_features(CMLIRSupport PUBLIC cxx_std_17)
add_library(CMLIR::Support ALIAS CMLIRSupport)
