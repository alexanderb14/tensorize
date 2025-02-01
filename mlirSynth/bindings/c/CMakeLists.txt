add_mlir_public_c_api_library(MLIRSYNTHCAPI
  MlirSynthCApi.cc
  MlirSynthPasses.cc

  LINK_LIBS PUBLIC
  common
  execution
  translation
  transform
  synthesis
)