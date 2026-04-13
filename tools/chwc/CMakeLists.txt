add_clang_executable(chwc
  driver.cpp
)

target_link_libraries(chwc PRIVATE
  cmlir

  # LLVM Libraries

  # MLIR Libraries
  MLIRSupport
  MLIRIR

  # Clang Libraries
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

add_dependencies(chwc MLIRCMLIROpsIncGen MLIRCMLIRTransformsIncGen)
