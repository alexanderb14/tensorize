#!/bin/bash

set -x
set -e

# Preparations
# ####
# Parse build type
BUILD_DIR=build
BUILD_TYPE=
if [ "$1" == "--debug" ]; then
  BUILD_DIR=build_debug
  BUILD_TYPE=Debug
fi

# Build mlirSynth
# ####
# Autogenerate arg tuple construction source file.
python3 build_tools/gen_CartesianProduct.py \
  --max_operands 3 \
  --max_attributes 3 \
  --max_regions 2 \
  --output mlirSynth/synthesis/CartesianProduct.cc
clang-format -i mlirSynth/synthesis/CartesianProduct.cc --style=file

# Configure mlirSynth build.
mkdir -p $BUILD_DIR
pushd $BUILD_DIR
FLAGS=".. \
  -GNinja \
  -DCMAKE_BUILD_TYPE=$BUILD_TYPE \
  -DMLIR_DIR=${PWD}/../deps/llvm-project/${BUILD_DIR}/lib/cmake/mlir \
  -DMHLO_DIR=${PWD}/../deps/mlir-hlo/${BUILD_DIR}/cmake/modules/CMakeFiles \
  -DCMAKE_C_COMPILER=clang \
  -DCMAKE_CXX_COMPILER=clang++ \
  -DCMAKE_EXPORT_COMPILE_COMMANDS=ON "
if [[ "$OSTYPE" == "linux-gnu"* ]]; then
  FLAGS="$FLAGS -DLLVM_ENABLE_LLD=ON"
fi
cmake $FLAGS

popd

# Generate Grammar from tablegen files.
pushd $BUILD_DIR
cmake --build . --target grammar-extractor
popd
TD_OPS="deps/mlir-hlo/stablehlo/stablehlo/dialect/ChloOps.td \
deps/mlir-hlo/stablehlo/stablehlo/dialect/StablehloOps.td \
deps/llvm-project/mlir/include/mlir/Dialect/Linalg/IR/LinalgOps.td \
deps/llvm-project/mlir/include/mlir/Dialect/Linalg/IR/LinalgStructuredOps.td"
TD_INCLUDES="-Ideps/mlir-hlo \
-Ideps/mlir-hlo/include \
-Ideps/mlir-hlo/include/mlir-hlo/Dialect/mhlo/IR \
-Ideps/mlir-hlo/stablehlo \
-Ideps/llvm-project/llvm/include \
-Ideps/llvm-project/mlir/include \
-Ideps/llvm-project/build/include \
-Ideps/llvm-project/build/tools/mlir/include \
-Ideps/llvm-project/mlir/include/mlir/Dialect/Linalg/IR"
cat $TD_OPS | ./build/bin/grammar-extractor $TD_INCLUDES \
  -gen-grammar-decls -o mlirSynth/synthesis/Grammar.h
cat $TD_OPS | ./build/bin/grammar-extractor $TD_INCLUDES \
  -gen-grammar-defs -o mlirSynth/synthesis/Grammar.cc

# Build mlirSynth.
pushd $BUILD_DIR
cmake --build .
popd

# Merge all compile_commands.json files, so that clangd can find them.
jq -s 'map(.[])' deps/llvm-project/build/compile_commands.json \
  deps/mlir-hlo/build/compile_commands.json \
  build/compile_commands.json \
  > compile_commands.json
