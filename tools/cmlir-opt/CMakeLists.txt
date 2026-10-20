add_llvm_executable(cmlir-opt driver.cpp)
llvm_update_compile_flags(cmlir-opt)
target_link_libraries(cmlir-opt PRIVATE
  CMLIR::Dialect
  CMLIR::Transforms
  CMLIR::Pipelines
  MLIROptLib
  MLIRRegisterAllDialects
  MLIRRegisterAllPasses
  MLIRRegisterAllExtensions
)
mlir_check_all_link_libraries(cmlir-opt)
