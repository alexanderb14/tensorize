#ifndef IRSYNTH_VALIDATION_TESTINGVALIDATOR_H
#define IRSYNTH_VALIDATION_TESTINGVALIDATOR_H

#include "mlir/IR/BuiltinOps.h"

bool testValidate(mlir::ModuleOp lhsModule, mlir::ModuleOp rhsModule,
                  bool printArgsAndResults = false, bool printResults = false);

#endif // IRSYNTH_VALIDATION_TESTINGVALIDATOR_H
