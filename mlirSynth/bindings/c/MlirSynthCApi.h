#ifndef MLIRSYNTH_C_API_H
#define MLIRSYNTH_C_API_H

#include <stdbool.h>
#include <stdint.h>

#include "mlir-c/IR.h"
#include "mlir-c/Support.h"

#include "mlir/CAPI/Wrap.h"
#include "synthesis/Options.h"

#ifdef __cplusplus
extern "C" {
#endif

//===----------------------------------------------------------------------===//
/// Opaque type declarations.
///
/// Types are exposed to C bindings as structs containing opaque pointers. They
/// are not supposed to be inspected from C. This allows the underlying
/// representation to change without affecting the API users. The use of structs
/// instead of typedefs enables some type safety as structs are not implicitly
/// convertible to each other.
///
/// Instances of these types may or may not own the underlying object (most
/// often only point to an IR fragment without owning it). The ownership
/// semantics is defined by how an instance of the type was obtained.

//===----------------------------------------------------------------------===//

#define DEFINE_C_API_STRUCT(name, storage)                                     \
  struct name {                                                                \
    storage *ptr;                                                              \
  };                                                                           \
  typedef struct name name

DEFINE_C_API_STRUCT(StdString, void);
DEFINE_C_API_PTR_METHODS(StdString, std::string);

MLIR_CAPI_EXPORTED void registerDialects(MlirContext context);
MLIR_CAPI_EXPORTED void lowerCHLOToAffine(MlirModule module,
                                          bool expandArithOps);
MLIR_CAPI_EXPORTED void emitPython(MlirModule module, StdString pyCode);
MLIR_CAPI_EXPORTED void predictGuideOps(MlirOperation op, StdString ops);
MLIR_CAPI_EXPORTED void predictConstants(MlirOperation op, StdString constants);
MLIR_CAPI_EXPORTED void
enumerateOneOp(MlirStringRef *operations, unsigned numOperations,
               MlirType *argTypes, unsigned numArgTypes, MlirType returnType,
               SynthesisOptions options, void *resultModules);

#ifdef __cplusplus
}
#endif

#endif // MLIRSYNTH_C_API_H