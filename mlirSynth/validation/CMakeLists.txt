add_mlir_library(validation
  CheckingValidator.cc
  TestingValidator.cc

  LINK_LIBS PUBLIC
  MLIRTargetCpp
)