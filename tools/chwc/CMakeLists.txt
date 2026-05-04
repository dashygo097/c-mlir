add_subdirectory(Runtime)

add_clang_executable(chwc
  Lib/Conversions/Types/Builtin.cpp
  Lib/Conversions/Types/Types.cpp

  Lib/Conversions/Exprs/Binary/Assign.cpp
  Lib/Conversions/Exprs/Binary/Pure.cpp
  Lib/Conversions/Exprs/Binary/Binary.cpp
  Lib/Conversions/Exprs/Unary/Unary.cpp
  Lib/Conversions/Exprs/Unary/IncDec.cpp
  Lib/Conversions/Exprs/Unary/AddrOf.cpp
  Lib/Conversions/Exprs/Literals.cpp
  Lib/Conversions/Exprs/DeclRef.cpp
  Lib/Conversions/Exprs/ImplicitCast.cpp
  Lib/Conversions/Exprs/Member.cpp
  Lib/Conversions/Exprs/CXXMemberCall.cpp
  Lib/Conversions/Exprs/ExprWithCleanups.cpp
  Lib/Conversions/Exprs/CXXBindTemporary.cpp
  Lib/Conversions/Exprs/MaterializeTemporary.cpp
  Lib/Conversions/Exprs/CXXConstruct.cpp
  Lib/Conversions/Exprs/CXXOperatorCall.cpp
  Lib/Conversions/Exprs/CXXFunctionalCast.cpp
  Lib/Conversions/Exprs/Exprs.cpp

  Lib/Conversions/Stmts/Compound.cpp
  Lib/Conversions/Stmts/Decl.cpp
  Lib/Conversions/Stmts/If.cpp
  Lib/Conversions/Stmts/Return.cpp
  Lib/Conversions/Stmts/Stmts.cpp

  Lib/Conversions/Decls/Function.cpp
  Lib/Conversions/Decls/CXXRecord.cpp

  Lib/Conversions/Hal/Module.cpp
  Lib/Conversions/Hal/Clock.cpp
  Lib/Conversions/Hal/Reset.cpp
  Lib/Conversions/Hal/Field.cpp

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
