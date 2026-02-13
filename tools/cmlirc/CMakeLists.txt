add_clang_executable(cmlirc
  Lib/Conversions/Types/Types.cpp

  Lib/Conversions/Exprs/Paren.cpp
  Lib/Conversions/Exprs/Literals.cpp
  Lib/Conversions/Exprs/Unary.cpp
  Lib/Conversions/Exprs/Binary.cpp
  Lib/Conversions/Exprs/DeclRef.cpp
  Lib/Conversions/Exprs/ImplicitCast.cpp
  Lib/Conversions/Exprs/Call.cpp
  Lib/Conversions/Exprs/ArraySubscript.cpp
  Lib/Conversions/Exprs/Exprs.cpp

  Lib/Conversions/Stmts/CF/If.cpp
  Lib/Conversions/Stmts/CF/For.cpp
  Lib/Conversions/Stmts/Return.cpp
  Lib/Conversions/Stmts/Stmts.cpp

  Lib/Conversions/Decls/Var.cpp
  Lib/Conversions/Decls/Function.cpp

  Lib/Conversions/Helpers.cpp

  Lib/Pragmas/PragmaHandler.cpp

  Lib/Context/ContextManager.cpp

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
