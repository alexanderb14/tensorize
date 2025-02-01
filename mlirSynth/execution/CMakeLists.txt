add_mlir_library(execution
  ArgUtils.cc
  ArrayUtils.cc
  Executor.cc
  Lowering.cc

  LINK_LIBS PUBLIC
  common
  MLIRExecutionEngine
  ${MHLO_EXPORTED_TARGETS}
)