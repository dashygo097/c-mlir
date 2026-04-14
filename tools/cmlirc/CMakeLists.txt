add_clang_executable(cmlirc
  Lib/Conversions/Types/Builtin.cpp
  Lib/Conversions/Types/Pointer.cpp
  Lib/Conversions/Types/Array.cpp
  Lib/Conversions/Types/Typedef.cpp
  Lib/Conversions/Types/Record.cpp
  Lib/Conversions/Types/Reference.cpp
  Lib/Conversions/Types/Types.cpp

  Lib/Conversions/Exprs/Binary/Assign.cpp
  Lib/Conversions/Exprs/Binary/Pure.cpp
  Lib/Conversions/Exprs/Binary/Binary.cpp
  Lib/Conversions/Exprs/Unary/AddrOf.cpp
  Lib/Conversions/Exprs/Unary/IncDec.cpp
  Lib/Conversions/Exprs/Unary/Unary.cpp
  Lib/Conversions/Exprs/Paren.cpp
  Lib/Conversions/Exprs/Literals.cpp
  Lib/Conversions/Exprs/DeclRef.cpp
  Lib/Conversions/Exprs/ImplicitCast.cpp
  Lib/Conversions/Exprs/CStyleCast.cpp
  Lib/Conversions/Exprs/InitList.cpp
  Lib/Conversions/Exprs/ArraySubscript.cpp
  Lib/Conversions/Exprs/Conditional.cpp
  Lib/Conversions/Exprs/Call.cpp
  Lib/Conversions/Exprs/CXXConstruct.cpp
  Lib/Conversions/Exprs/Member.cpp
  Lib/Conversions/Exprs/SideEffects.cpp
  Lib/Conversions/Exprs/UnaryOrTypeTrait.cpp
  Lib/Conversions/Exprs/CXXThis.cpp
  Lib/Conversions/Exprs/Exprs.cpp

  Lib/Conversions/Stmts/CF/If.cpp
  Lib/Conversions/Stmts/CF/Switch.cpp
  Lib/Conversions/Stmts/Loop/LoopUtils.cpp
  Lib/Conversions/Stmts/Loop/For.cpp
  Lib/Conversions/Stmts/Loop/While.cpp
  Lib/Conversions/Stmts/Loop/Do.cpp
  Lib/Conversions/Stmts/Loop/Break.cpp
  Lib/Conversions/Stmts/Loop/Continue.cpp
  Lib/Conversions/Stmts/Return.cpp
  Lib/Conversions/Stmts/Stmts.cpp

  Lib/Conversions/Decls/Var.cpp
  Lib/Conversions/Decls/Function.cpp
  Lib/Conversions/Decls/Record.cpp
  Lib/Conversions/Decls/CXXMethod.cpp

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
  MLIRArithTransforms
  MLIRFuncDialect
  MLIRMemRefDialect
  MLIRMemRefTransforms
  MLIRSCFDialect 
  MLIRSCFTransforms
  MLIRMathDialect
  MLIRVectorDialect
  MLIRTransforms
  MLIRTransformUtils
  MLIRFuncInlinerExtension
  MLIRLLVMIRTransforms

  # Clang Libraries
  clangSupport
  clangTooling
)

add_dependencies(cmlirc MLIRCMLIROpsIncGen MLIRCMLIRTransformsIncGen)
