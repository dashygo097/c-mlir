add_clang_executable(cmlirc
  Lib/MLIRContextManager.cpp
  Lib/TypeConverter.cpp
  Lib/ASTVisitor.cpp

  driver.cpp
)

target_link_libraries(cmlirc PRIVATE
  cmlir 

  # LLVM Libraries

  # MLIR Libraries
  MLIRSupport
  MLIRIR
  MLIRLLVMDialect
  MLIRArithDialect
  MLIRFuncDialect
  MLIRMemRefDialect
  MLIRSCFDialect 

  # Clang Libraries
  clangSupport
  clangTooling
)

add_dependencies(cmlirc MLIRCMLIROpsIncGen)
