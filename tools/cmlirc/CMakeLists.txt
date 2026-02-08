add_clang_executable(cmlirc
  Lib/Conversions/Types.cpp
  Lib/Conversions/UnaryExprs.cpp
  Lib/Conversions/BinaryExprs.cpp
  Lib/Conversions/Exprs.cpp

  Lib/ContextManager.cpp
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
  MLIRTransforms

  # Clang Libraries
  clangSupport
  clangTooling
)

add_dependencies(cmlirc MLIRCMLIROpsIncGen MLIRCMLIRTransformsIncGen)
