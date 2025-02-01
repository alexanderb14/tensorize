# Needs to be seperate to be compiled with -frtti
add_library(cartesian_product
  CartesianProduct.cc)

target_link_libraries(cartesian_product
  PUBLIC
  range_v3)

add_mlir_library(synthesis
  Generators.cc
  Candidate.cc
  CandidateStore.cc
  Synthesizer.cc
  Guide.cc
  Grammar.cc
  ProcessingStatus.cc
  Spec.cc
  Stats.cc

  LINK_LIBS PUBLIC
  ${SYNTH_LIBS}
  common
  cartesian_product
  analysis
)