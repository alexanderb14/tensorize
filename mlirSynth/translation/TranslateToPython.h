#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Value.h"
#include "llvm/ADT/ScopedHashTable.h"
#include "llvm/Support/raw_ostream.h"
#include <stack>

mlir::LogicalResult translateToPython(mlir::Operation *op, mlir::raw_ostream &os,
                                    bool declareVariablesAtTop);