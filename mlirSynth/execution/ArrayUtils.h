#ifndef IRSYNTH_ARRAYUTILS_H
#define IRSYNTH_ARRAYUTILS_H

#include "llvm/ADT/ArrayRef.h"

#include <cstdint>

void printArray(double *arr, llvm::ArrayRef<int64_t> shape,
                llvm::raw_ostream &os);
double hashArray(double *arr, llvm::ArrayRef<int64_t> shape);
bool areArraysEqual(double *arr1, double *arr2, llvm::ArrayRef<int64_t> shape);
double getDelta(double *arr1, double *arr2, llvm::ArrayRef<int64_t> shape);

#endif // IRSYNTH_ARRAYUTILS_H
