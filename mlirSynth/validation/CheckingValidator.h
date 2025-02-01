#ifndef IRSYNTH_VALIDATION_CHECKINGVALIDATOR_H
#define IRSYNTH_VALIDATION_CHECKINGVALIDATOR_H

#include "mlir/IR/BuiltinOps.h"

bool checkValidate(mlir::ModuleOp lhsModule, mlir::ModuleOp rhsModule,
                   bool printArgsAndResults = false, bool printResults = false);

#endif // IRSYNTH_VALIDATION_CHECKINGVALIDATOR_H
