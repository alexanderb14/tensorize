# Specifies that all MLIR packages are co-located under the `mlir_synth`
# top level package (the API has been embedded in a relocatable way).
# TODO: Add an upstream cmake param for this vs having a global here.
add_compile_definitions("MLIR_PYTHON_PACKAGE_PREFIX=mlir_synth.")


################################################################################
# Sources
################################################################################

declare_mlir_python_sources(SynthPythonSources)

declare_mlir_dialect_python_bindings(
  ADD_TO_PARENT SynthPythonSources
  ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/mlir_synth"
#  TD_FILE dialects/SynthOps.td
  SOURCES
    synth.py
  DIALECT_NAME synth)

declare_mlir_python_extension(SynthPythonSources.Extension
  MODULE_NAME _synth
  ADD_TO_PARENT SynthPythonSources
  SOURCES
    MlirSynthExtension.cc

  PRIVATE_LINK_LIBS
    LLVMSupport
  EMBED_CAPI_LINK_LIBS
    MLIRSYNTHCAPI

    MLIRCAPIDebug
    MLIRCAPIIR
    MLIRCAPIInterfaces

#  PRIVATE_LINK_LIBS
#    execution
)

################################################################################
# Common CAPI
################################################################################
add_mlir_python_common_capi_library(SynthPythonCAPI
  INSTALL_COMPONENT SynthPythonModules
  INSTALL_DESTINATION python_packages/synth/mlir_synth/_mlir_libs
  OUTPUT_DIRECTORY "${MLIR_BINARY_DIR}/python_packages/synth/mlir_synth/_mlir_libs"
  RELATIVE_INSTALL_ROOT "../../../.."
  DECLARED_SOURCES
    SynthPythonSources
    SynthPythonSources.Extension
    MLIRPythonExtension.RegisterEverything
    MLIRPythonSources.Core
)

################################################################################
# Instantiation of all Python modules
################################################################################

add_mlir_python_modules(SynthPythonModules
  ROOT_PREFIX "${MLIR_BINARY_DIR}/python_packages/synth/mlir_synth"
  INSTALL_PREFIX "python_packages/synth/mlir_synth"
  DECLARED_SOURCES
    SynthPythonSources
    MLIRPythonExtension.RegisterEverything
    MLIRPythonSources

    #MLIRHLOPythonSources
    #MLIRHLOPythonExtensions
  COMMON_CAPI_LINK_LIBS
    SynthPythonCAPI
    #MLIRHLOCAPI
  )
