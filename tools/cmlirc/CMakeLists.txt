add_clang_executable(cmlirc
  Lib/ValueCategory.cpp
  Lib/ASTVisitor.cpp

  cmlirc.cpp
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

  # Clang Libraries
  clangSupport
  clangTooling
)

add_dependencies(cmlirc MLIRCMLIROpsIncGen)
