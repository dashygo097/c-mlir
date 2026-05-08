add_subdirectory(Runtime)

add_clang_executable(chwc
  Lib/Conversions/Types/Builtin.cpp
  Lib/Conversions/Types/Types.cpp

  Lib/Conversions/Exprs/ArraySubscript.cpp
  Lib/Conversions/Exprs/Assign.cpp
  Lib/Conversions/Exprs/Binary.cpp
  Lib/Conversions/Exprs/CXXBindTemporary.cpp
  Lib/Conversions/Exprs/CXXConstruct.cpp
  Lib/Conversions/Exprs/CXXFunctionalCast.cpp
  Lib/Conversions/Exprs/CXXMemberCall.cpp
  Lib/Conversions/Exprs/OperatorCall.cpp
  Lib/Conversions/Exprs/Call.cpp
  Lib/Conversions/Exprs/DeclRef.cpp
  Lib/Conversions/Exprs/ExprWithCleanups.cpp
  Lib/Conversions/Exprs/Exprs.cpp
  Lib/Conversions/Exprs/ImplicitCast.cpp
  Lib/Conversions/Exprs/Literals.cpp
  Lib/Conversions/Exprs/MaterializeTemporary.cpp
  Lib/Conversions/Exprs/Member.cpp
  Lib/Conversions/Exprs/Pure.cpp
  Lib/Conversions/Exprs/Unary.cpp

  Lib/Conversions/Stmts/Compound.cpp
  Lib/Conversions/Stmts/Decl.cpp
  Lib/Conversions/Stmts/For.cpp
  Lib/Conversions/Stmts/If.cpp
  Lib/Conversions/Stmts/Return.cpp
  Lib/Conversions/Stmts/Stmts.cpp

  Lib/Conversions/Decls/CXXRecord.cpp
  Lib/Conversions/Decls/Field.cpp
  Lib/Conversions/Decls/Function.cpp

  Lib/Context/ContextManager.cpp
  driver.cpp
)

target_link_libraries(chwc PRIVATE
  chwc-runtime

  cmlir

  # MLIR
  MLIRSupport
  MLIRIR

  # Clang
  clangSupport
  clangTooling

  # CIRCT
  CIRCTSupport
  CIRCTHW
  CIRCTComb
  CIRCTSeq
  CIRCTSV
  CIRCTFirtool
)

target_compile_definitions(chwc PRIVATE
  CHWC_RUNTIME_INCLUDE_DIR="${CMAKE_CURRENT_SOURCE_DIR}/Runtime/include"
)

add_dependencies(chwc
  MLIRCMLIROpsIncGen
  MLIRCMLIRTransformsIncGen
)
