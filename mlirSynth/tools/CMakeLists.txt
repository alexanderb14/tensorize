# Grammar Extractor
set(LLVM_LINK_COMPONENTS TableGen)
add_llvm_executable(grammar-extractor
  GrammarExtractor.cc)
target_link_libraries(grammar-extractor
  PRIVATE MLIRIR
          MLIRTableGen
          MLIRTblgenLib
          MLIRSupport)

# Lower CHLO to Affine tool
add_llvm_executable(lower-chlo-to-affine
  LowerCHLOToAffine.cc)
target_include_directories(lower-chlo-to-affine
  PRIVATE ${CMAKE_CURRENT_LIST_DIR})
target_link_libraries(lower-chlo-to-affine
  PRIVATE analysis
          synthesis
          execution
          transform)
llvm_update_compile_flags(lower-chlo-to-affine)


# Transform Tool
add_llvm_executable(opt
  Opt.cc)
target_include_directories(opt
  PRIVATE ${CMAKE_CURRENT_LIST_DIR}
          ${MHLO_INCLUDE_DIRS})
target_link_libraries(opt
  PRIVATE ${MHLO_EXPORTED_TARGETS}
          analysis
          transform)

# Synthesizer
add_llvm_executable(synthesizer
  Synthesizer.cc)
target_include_directories(synthesizer
  PRIVATE ${CMAKE_CURRENT_LIST_DIR})
target_link_libraries(synthesizer
  PRIVATE analysis
          common
          synthesis
          execution
          transform)
llvm_update_compile_flags(synthesizer)

# Test Executor
add_llvm_executable(test-executor
  TestExecutor.cc)
target_include_directories(test-executor
  PRIVATE ${CMAKE_CURRENT_LIST_DIR})
target_link_libraries(test-executor
  PRIVATE analysis
          common
          synthesis
          execution
          transform
          validation)
llvm_update_compile_flags(test-executor)

# Affine to Python Translator
add_llvm_executable(translate-affine-to-python
  TranslateAffineToPython.cc)
target_include_directories(translate-affine-to-python
  PRIVATE ${CMAKE_CURRENT_LIST_DIR})
target_link_libraries(translate-affine-to-python
  PUBLIC  translation
          ${dialect_libs})

llvm_update_compile_flags(translate-affine-to-python)
