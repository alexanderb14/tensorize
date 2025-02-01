add_library(analysis
  PolyhedralAnalysis.cc
  isl/isl_helper.cc)
target_include_directories(analysis
  PUBLIC ${CMAKE_SOURCE_DIR}/deps/isl-0.25/include)
target_link_libraries(analysis
  PRIVATE ${dialect_libs}
          ${conversion_libs}
          ${test_libs}
          MLIRAffineAnalysis
          MLIRAnalysis
          MLIRDialect
          MLIROptLib
          MLIRParser
          MLIRPass
          MLIRTransforms
          MLIRTransformUtils
          MLIRSupport
          MLIRIR
          ${CMAKE_SOURCE_DIR}/deps/isl-0.25/.libs/libisl.a
          ${GMP_LIB})
llvm_update_compile_flags(analysis)
target_compile_options(analysis
  PRIVATE -Wno-missing-field-initializers)