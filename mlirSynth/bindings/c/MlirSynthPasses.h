#ifndef MLIRSYNTH_PASSES_H
#define MLIRSYNTH_PASSES_H

#include "mlir-c/IR.h"
#include "mlir-c/Support.h"

#ifdef __cplusplus
extern "C" {
#endif

MLIR_CAPI_EXPORTED void registerMlirSynthPasses();

#ifdef __cplusplus
}
#endif

#endif  // MLIRSYNTH_PASSES_H
