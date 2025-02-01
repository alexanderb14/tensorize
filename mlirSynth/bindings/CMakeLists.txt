include(MLIRDetectPythonEnv)
mlir_configure_python_dev_packages()

include(AddMLIRPython)

add_subdirectory(c)
add_subdirectory(python)