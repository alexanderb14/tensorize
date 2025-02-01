#!/bin/bash

set -x
set -e

# Preparations
# ####
# Parse build type
BUILD_DIR=build
BUILD_TYPE=RelWithDebInfo
if [ "$1" == "--debug" ]; then
  BUILD_DIR=build_debug
  BUILD_TYPE=Debug
fi

# Prepare deps directory
mkdir -p deps
pushd deps

# MLIR-HLO and MLIR
# ####
# Pull
if [ ! -d "mlir-hlo" ]; then
  git clone https://github.com/tensorflow/mlir-hlo.git
  pushd mlir-hlo
  git checkout abf4e4c1095fe17611437c3bed108dc60c9d92e0
  popd
fi

if [ ! -d "llvm-project" ]; then
  git clone https://github.com/llvm/llvm-project.git
  pushd llvm-project
  git checkout $(cat ../mlir-hlo/build_tools/llvm_version.txt)
  git am < ../../build_tools/patches/llvm/add-trait-verification-function.patch
  git am < ../../build_tools/patches/llvm/enable-emit-c-for-more-ops.patch
  popd
fi

# Build
pushd llvm-project
mkdir -p $BUILD_DIR
pushd $BUILD_DIR

FLAGS="../llvm \
  -GNinja \
  -DLLVM_INSTALL_UTILS=ON \
  -DLLVM_ENABLE_PROJECTS=mlir \
  -DLLVM_INCLUDE_TOOLS=ON \
  -DLLVM_ENABLE_BINDINGS=OFF \
  -DLLVM_BUILD_TOOLS=OFF \
  -DLLVM_INCLUDE_TESTS=OFF \
  -DMLIR_ENABLE_BINDINGS_PYTHON=ON \
  -DCMAKE_BUILD_TYPE=$BUILD_TYPE \
  -DLLVM_PARALLEL_LINK_JOBS=1 \
  -DCMAKE_EXPORT_COMPILE_COMMANDS=ON "
if [[ "$OSTYPE" == "linux-gnu"* ]]; then
  FLAGS="$FLAGS -DLLVM_ENABLE_LLD=ON \
    -DCMAKE_C_COMPILER=clang \
    -DCMAKE_CXX_COMPILER=clang++ \
    -DLLVM_TARGETS_TO_BUILD=X86"
elif [[ "$OSTYPE" == "darwin"* ]]; then
  FLAGS="$FLAGS -DLLVM_TARGETS_TO_BUILD=AArch64"
fi
cmake $FLAGS

cmake --build .
popd
popd

# Build MLIR-HLO.
pushd mlir-hlo
mkdir -p $BUILD_DIR
pushd $BUILD_DIR

FLAGS=".. \
  -GNinja \
  -DCMAKE_BUILD_TYPE=$BUILD_TYPE \
  -DLLVM_ENABLE_ASSERTIONS=On \
  -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
  -DMLIR_DIR=${PWD}/../../llvm-project/${BUILD_DIR}/lib/cmake/mlir "
if [[ "$OSTYPE" == "linux-gnu"* ]]; then
  FLAGS="$FLAGS -DLLVM_ENABLE_LLD=ON \
    -DCMAKE_C_COMPILER=clang \
    -DCMAKE_CXX_COMPILER=clang++"
fi
cmake $FLAGS

cmake --build .
popd
popd

# ISL
# ####
# Pull
wget -P /tmp https://libisl.sourceforge.io/isl-0.25.tar.gz
tar -xf /tmp/isl-0.25.tar.gz
rm /tmp/isl-0.25.tar.gz

# Build
pushd isl-0.25
if [[ "$OSTYPE" == "linux-gnu"* ]]; then
  CC=clang CXX=clang++ ./configure --with-pic
elif [[ "$OSTYPE" == "darwin"* ]]; then
  CFLAGS=-I/opt/homebrew/opt/gmp/include LDFLAGS=-L/opt/homebrew/opt/gmp/lib \
  CC=clang CXX=clang++ ./configure --with-pic
fi
make -j$(nproc)
popd

# Indicators
# ####
# Pull
if [ ! -d "indicators" ]; then
  git clone https://github.com/p-ranav/indicators
  pushd indicators
  git checkout ef71abd9bc7254f7734fa84d5b1c336be2deb9f7
  popd
fi

# Ranges V3
# ####
# Pull
if [ ! -d "range-v3" ]; then
  git clone https://github.com/ericniebler/range-v3
  pushd range-v3
  #git checkout ef71abd9bc7254f7734fa84d5b1c336be2deb9f7
  popd
fi

# Build
pushd indicators
python3 utils/amalgamate/amalgamate.py -c single_include.json -s .
popd
