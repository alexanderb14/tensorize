#include "bindings/c/MlirSynthPasses.h"

#include "transforms/Passes.h"
#include "mlir-hlo/Dialect/mhlo/transforms/passes.h"
#include "mlir-hlo/Transforms/passes.h"

void registerMlirSynthPasses() {
    registerMlirSynthAllPasses(); 
    mlir::mhlo::registerAllMhloPasses();
}
