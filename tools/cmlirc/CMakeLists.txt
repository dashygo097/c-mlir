add_clang_executable(cmlirc
  Lib/ASTVisitor.cpp

  cmlirc.cpp
)

target_link_libraries(cmlirc PRIVATE
  cmlir 

  # LLVM Libraries

  # Clang Libraries
  clangTooling
)

add_dependencies(cmlirc MLIRCMLIROpsIncGen)
