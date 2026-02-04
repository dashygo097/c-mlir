add_clang_executable(cmlirc
  cmlirc.cpp
)

target_link_libraries(cmlirc PRIVATE
  cmlir 
)

add_dependencies(cmlirc MLIRCMLIROpsIncGen)
