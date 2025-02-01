cmake_minimum_required(VERSION 3.15.0)

if(POLICY CMP0068)
  cmake_policy(SET CMP0068 NEW)
  set(CMAKE_BUILD_WITH_INSTALL_NAME_DIR ON)
endif()

if(POLICY CMP0075)
  cmake_policy(SET CMP0075 NEW)
endif()

if(POLICY CMP0077)
  cmake_policy(SET CMP0077 NEW)
endif()

project(synth-ir LANGUAGES CXX C)
set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD 17)

# Dependencies
# ##############################################################################
# MLIR
find_package(MLIR REQUIRED CONFIG)

message(STATUS "using mlirconfig.cmake in: ${MLIR_DIR}")
message(STATUS "using llvmconfig.cmake in: ${LLVM_DIR}")
message(STATUS "using mhloconfig.cmake in: ${MHLO_DIR}")
set(LLVM_RUNTIME_OUTPUT_INTDIR ${CMAKE_BINARY_DIR}/bin)
set(LLVM_LIBRARY_OUTPUT_INTDIR ${CMAKE_BINARY_DIR}/lib)
list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")
set(MLIR_BINARY_DIR ${CMAKE_BINARY_DIR})

include(TableGen)
include(AddLLVM)
include(AddMLIR)
include(HandleLLVMOptions)

include_directories(${LLVM_INCLUDE_DIRS})
include_directories(${MLIR_INCLUDE_DIRS})

link_directories(${LLVM_BUILD_LIBRARY_DIR})
add_definitions(${LLVM_DEFINITIONS})

# MLIR-HLO
find_package(MHLO)
list(APPEND CMAKE_MODULE_PATH "${MHLO_CMAKE_DIR}")
list(APPEND MHLO_INCLUDE_DIRS "${CMAKE_CURRENT_SOURCE_DIR}/deps/mlir-hlo")
list(APPEND MHLO_INCLUDE_DIRS "${CMAKE_CURRENT_SOURCE_DIR}/deps/mlir-hlo/include")
list(APPEND MHLO_INCLUDE_DIRS "${CMAKE_CURRENT_SOURCE_DIR}/deps/mlir-hlo/lhlo")
list(APPEND MHLO_INCLUDE_DIRS "${CMAKE_CURRENT_SOURCE_DIR}/deps/mlir-hlo/stablehlo")
list(APPEND MHLO_INCLUDE_DIRS "${CMAKE_CURRENT_SOURCE_DIR}/deps/mlir-hlo/build")
list(APPEND MHLO_INCLUDE_DIRS "${CMAKE_CURRENT_SOURCE_DIR}/deps/mlir-hlo/build/lhlo")
list(APPEND MHLO_INCLUDE_DIRS
     "${CMAKE_CURRENT_SOURCE_DIR}/deps/mlir-hlo/build/stablehlo")
#include_directories(${MHLO_INCLUDE_DIRS})

# Boost
find_package(
  Boost 1.74
  COMPONENTS graph
  REQUIRED)

# Indicators
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/deps/indicators/single_include)

# ISL
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/deps/isl-0.25/include)

# Range v3
add_library(range_v3 INTERFACE IMPORTED)
set_target_properties(range_v3 PROPERTIES
    INTERFACE_INCLUDE_DIRECTORIES ${CMAKE_CURRENT_SOURCE_DIR}/deps/range-v3/include)

# GMP
if(UNIX AND NOT APPLE)
  set(GMP_LIB "/usr/lib/x86_64-linux-gnu/libgmp.so")
elseif(APPLE)
  set(GMP_LIB "/opt/homebrew/opt/gmp/lib/libgmp.a")
endif()

# MlirSynth
# ##############################################################################
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/mlirSynth)
include_directories(${CMAKE_CURRENT_BINARY_DIR}/mlirSynth)

add_subdirectory(mlirSynth)
