#ifndef IRSYNTH_UTILS_H
#define IRSYNTH_UTILS_H

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/OwningOpRef.h"
#include "llvm/ADT/Optional.h"

template <typename T>
class [[nodiscard]] SuccessOr : public llvm::Optional<T> {
public:
  SuccessOr(mlir::LogicalResult result) {
    assert(succeeded(result) &&
           "success should be constructed with an instance of 'T'");
  }
  SuccessOr() : SuccessOr(mlir::success()) {}
  SuccessOr(T &&y) : llvm::Optional<T>(std::forward<T>(y)) {}
  SuccessOr(const T &y) : llvm::Optional<T>(y) {}
  template <typename U,
            std::enable_if_t<std::is_constructible<T, U>::value> * = nullptr>
  SuccessOr(const SuccessOr<U> &other)
      : llvm::Optional<T>(succeeded(other) ? llvm::Optional<T>()
                                           : llvm::Optional<T>(*other)) {}

  operator mlir::LogicalResult() const { return failure(this->has_value()); }

private:
  using llvm::Optional<T>::operator bool;
  using llvm::Optional<T>::has_value;
};

mlir::OwningOpRef<mlir::ModuleOp> createModule(mlir::MLIRContext &ctx,
                                               mlir::func::FuncOp *function);
mlir::OwningOpRef<mlir::ModuleOp> createModule(mlir::MLIRContext &ctx,
                                               mlir::Region *region);

#endif // IRSYNTH_UTILS_H
