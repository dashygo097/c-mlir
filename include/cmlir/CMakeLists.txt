add_mlir_dialect(CMLIROps cmlir)

add_subdirectory(Transforms)

add_custom_target(header DEPENDS MLIRCMLIROpsIncGen MLIRCMLIRTransformsIncGen)
