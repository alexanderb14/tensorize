#ifndef IRSYNTH_GUIDE_H
#define IRSYNTH_GUIDE_H

#include "mlir/IR/Operation.h"

std::vector<std::string> predictOps(std::vector<std::string> &supportedOps,
                                    mlir::Operation *op);
std::vector<float> predictConstants(mlir::Operation *op);

#endif // IRSYNTH_GUIDE_H
