#include "transforms/Passes.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Pass/AnalysisManager.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"

using namespace mlir;

using SizeMap = llvm::DenseMap<int64_t, int64_t>;
using Shape = ::llvm::SmallVector<int64_t>;

namespace mlir {

#define GEN_PASS_DEF_CHANGESIZES
#include "transforms/Passes.h.inc"

namespace {

struct ChangeSizesPass : public impl::ChangeSizesBase<ChangeSizesPass> {
  void runOnOperation() override;
};

} // namespace
} // namespace mlir

int nextPrime(int n) {
  if (n <= 2)
    return 2;
  if (n <= 3)
    return 3;
  if (n % 2 == 0 || n % 3 == 0)
    return nextPrime(n + 1);
  for (int i = 5; i * i <= n; i = i + 6)
    if (n % i == 0 || n % (i + 2) == 0)
      return nextPrime(n + 1);
  return n;
}

llvm::SetVector<int64_t> getMemrefAndTensorTypeDimensions(ModuleOp &moduleOp) {
  bool debug = false;

  llvm::SetVector<MemRefType> memrefTypes;
  llvm::SetVector<RankedTensorType> tensorTypes;
  moduleOp->walk([&](Operation *op) {
    // - Collect from operation operands.
    for (auto operand : op->getOperands()) {
      if (operand.getType().isa<MemRefType>())
        memrefTypes.insert(operand.getType().cast<MemRefType>());
      if (operand.getType().isa<RankedTensorType>())
        tensorTypes.insert(operand.getType().cast<RankedTensorType>());
    }
    // - Collect from operation results.
    for (auto type : op->getResultTypes()) {
      if (type.isa<MemRefType>())
        memrefTypes.insert(type.cast<MemRefType>());
      if (type.isa<RankedTensorType>())
        tensorTypes.insert(type.cast<RankedTensorType>());
    }
  });

  if (debug) {
    llvm::outs() << "Collected memref types:\n";
    for (auto type : memrefTypes)
      llvm::errs() << type << "\n";
    llvm::outs() << "Collected tensor types:\n";
    for (auto type : tensorTypes)
      llvm::errs() << type << "\n";
  }

  // Collect all dimensions.
  llvm::SetVector<int64_t> dimensions;
  for (auto type : memrefTypes)
    for (auto dim : type.getShape())
      dimensions.insert(dim);
  for (auto type : tensorTypes)
    for (auto dim : type.getShape())
      dimensions.insert(dim);

  if (debug) {
    llvm::outs() << "Collected dimensions:\n";
    for (auto dim : dimensions)
      llvm::errs() << dim << "\n";
  }

  return dimensions;
}

SizeMap getMinifedSizeMap(ModuleOp &moduleOp, std::string &sizeMode) {
  bool debug = false;

  // Memref types.
  auto dimensions = getMemrefAndTensorTypeDimensions(moduleOp);

  // Sort them.
  Shape sortedDimensions;
  for (auto dim : dimensions)
    sortedDimensions.push_back(dim);
  std::sort(sortedDimensions.begin(), sortedDimensions.end());

  if (debug) {
    llvm::outs() << "Sorted dimensions:\n";
    for (auto dim : sortedDimensions)
      llvm::errs() << dim << "\n";
  }

  // Create a mapping from sorted dimensions to their minified values. Minified
  // values are prime numbers.
  SizeMap minifiedDimensions;
  int64_t minifiedValue = 3;
  for (auto dim : sortedDimensions) {
    if (sizeMode == "Primes") {
      minifiedDimensions[dim] = minifiedValue;
      minifiedValue = nextPrime(minifiedValue + 1);
    } else if (sizeMode == "Uniform") {
      minifiedDimensions[dim] = 4;
    } else {
      llvm::outs() << "Unknown size mode: " << sizeMode << "\n";
      assert(false && "Unknown size mode");
    }
  }

  if (debug) {
    llvm::outs() << "Minified dimensions:\n";
    for (auto dim : dimensions)
      llvm::errs() << dim << " -> " << minifiedDimensions[dim] << "\n";
  }

  return minifiedDimensions;
}

Shape resizeShape(::llvm::ArrayRef<int64_t> oldShape, SizeMap newSizes) {
  Shape newShape;

  for (auto dim : oldShape) {
    long newDim;
    if (newSizes.count(dim) == 0)
      newDim = dim;
    else
      newDim = newSizes[dim];
    newShape.push_back(newDim);
  }

  return newShape;
}

template <typename OpTy>
void changeTypeSizes(func::FuncOp &func, SizeMap &newSizes) {
  // In function signature.
  auto type = func.getFunctionType();
  // - Change memref types in function arguments.
  llvm::SmallVector<Type> newArgTypes;
  for (auto argType : type.getInputs()) {
    if (argType.isa<OpTy>()) {
      auto type = argType.cast<OpTy>();
      Shape newShape = resizeShape(type.getShape(), newSizes);
      auto newType = OpTy::get(newShape, type.getElementType());
      newArgTypes.push_back(newType);
    } else {
      newArgTypes.push_back(argType);
    }
  }

  // - Change memref types in function results.
  llvm::SmallVector<Type> newResultTypes;
  for (auto resultType : type.getResults()) {
    if (resultType.isa<OpTy>()) {
      auto type = resultType.cast<OpTy>();
      Shape newShape = resizeShape(type.getShape(), newSizes);
      auto newType = OpTy::get(newShape, type.getElementType());
      newResultTypes.push_back(newType);
    } else {
      newResultTypes.push_back(resultType);
    }
  }

  // - Set the new function type.
  auto newType =
      FunctionType::get(type.getContext(), newArgTypes, newResultTypes);
  func.setType(newType);

  // In the arguments of the body block of the function.
  auto &block = func.getBody().front();
  for (auto &arg : llvm::enumerate(block.getArguments())) {
    if (arg.value().getType().isa<OpTy>()) {
      auto type = arg.value().getType().cast<OpTy>();
      Shape newShape = resizeShape(type.getShape(), newSizes);
      auto newType = OpTy::get(newShape, type.getElementType());
      block.getArgument(arg.index()).setType(newType);
    }
  }

  // In operations.
  func->walk([&](Operation *op) {
    // - Change memref types in operation operands.
    for (auto operand : op->getOperands()) {
      if (operand.getType().isa<OpTy>()) {
        auto type = operand.getType().cast<OpTy>();
        Shape newShape = resizeShape(type.getShape(), newSizes);
        auto newType = OpTy::get(newShape, type.getElementType());
        operand.setType(newType);
      }
    }
    // - Change memref types in operation results.
    for (auto res : op->getResultTypes()) {
      if (res.isa<OpTy>()) {
        auto type = res.cast<OpTy>();
        Shape newShape = resizeShape(type.getShape(), newSizes);
        auto newType = OpTy::get(newShape, type.getElementType());
        type = newType;
      }
    }
  });
}

void changeConstantValues(func::FuncOp &func, SizeMap &newSizes) {
  func->walk([&](Operation *op) {
    // Check if op is a stablehlo.constant
    if (op->getName().getStringRef() == "stablehlo.constant") {
      // Get the value of the constant.
      auto value = op->getAttrOfType<DenseElementsAttr>("value");
      // Get the type of the constant.
      auto type = value.getType().cast<ShapedType>();
      // Get the shape of the constant.
      auto shape = type.getShape();
      // Resize the shape.
      auto newShape = resizeShape(shape, newSizes);
      // Create a new type with the new shape.
      auto newType = RankedTensorType::get(newShape, type.getElementType());
      // Create a new value with the new type.
      auto newValue = value.reshape(newType);
      // Set the new value.
      op->setAttr("value", newValue);
    }
    //    else if (op->getName().getStringRef() == "arith.constant") {
    //      // Change arith.constant of type index.
    //      if (auto indexAttr = op->getAttrOfType<IntegerAttr>("value")) {
    //        auto index = indexAttr.getInt();
    //        if (newSizes.count(index) > 0) {
    //          auto newIndex = newSizes[index];
    //          auto indexType = IntegerType::get(op->getContext(), 64);
    //          op->setAttr("value", IntegerAttr::get(indexType, newIndex));
    //        }
    //      }
    //    }
  });
}

void changeGlobals(ModuleOp &moduleOp, SizeMap &newSizes) {
  moduleOp->walk([&](Operation *op) {
    if (isa<memref::GlobalOp>(op)) {
      // Set the new type.
      auto globalOp = cast<memref::GlobalOp>(op);
      auto type = globalOp.getType();
      if (type.isa<MemRefType>()) {
        auto memrefType = type.cast<MemRefType>();
        Shape newShape = resizeShape(memrefType.getShape(), newSizes);
        auto newType = MemRefType::get(newShape, memrefType.getElementType());
        globalOp.setType(newType);
      }

      // Set the new initial value.
      auto value = globalOp->getAttrOfType<DenseElementsAttr>("initial_value");
      if (value) {
        auto type = value.getType().cast<ShapedType>();
        auto shape = type.getShape();
        auto newShape = resizeShape(shape, newSizes);
        auto newType = RankedTensorType::get(newShape, type.getElementType());
        auto newValue = value.reshape(newType);
        globalOp->setAttr("initial_value", newValue);
      }
    }
  });
}

void changeLoopBounds(func::FuncOp &func, SizeMap &newSizes) {
  bool debug = false;

  func->walk([&](Operation *op) {
    if (isa<AffineForOp>(op)) {
      auto forOp = cast<AffineForOp>(op);
      auto ubMap = forOp.getUpperBoundMap();
      llvm::SmallVector<AffineExpr> lbExprs;
      llvm::SmallVector<AffineExpr> ubExprs;
      llvm::SmallVector<AffineExpr> stepExprs;
      for (auto expr : ubMap.getResults()) {
        if (expr.isa<AffineDimExpr>()) {
          auto dimExpr = expr.cast<AffineDimExpr>();
          auto dim = dimExpr.getPosition();
          if (newSizes.count(dim) == 0)
            ubExprs.push_back(expr);
          else
            ubExprs.push_back(
                getAffineConstantExpr(newSizes[dim], op->getContext()));
        } else if (expr.isa<AffineConstantExpr>()) {
          auto dim = expr.cast<AffineConstantExpr>().getValue();
          if (newSizes.count(dim) == 0)
            ubExprs.push_back(expr);
          else
            ubExprs.push_back(
                getAffineConstantExpr(newSizes[dim], op->getContext()));
        } else if (expr.isa<AffineBinaryOpExpr>()) {
        } else {
          llvm::outs() << "expr type: " << (unsigned int)expr.getKind() << "\n";
          assert(false && "Unexpected expression type");
        }
      }
      auto ubMapNew = AffineMap::get(ubMap.getNumDims(), ubMap.getNumSymbols(),
                                     ubExprs, op->getContext());
      if (ubMapNew.getNumResults())
        forOp.setUpperBoundMap(ubMapNew);

      if (debug) {
        llvm::outs() << "Loop bounds: " << ubMap << "\n";
        llvm::outs() << "Minified loop bounds: " << ubMapNew << "\n";
      }
    }
  });
}

void annotateChangedSizes(func::FuncOp &func, SizeMap &newSizes) {
  std::string newSizesStr;
  bool first = true;
  for (auto dim : newSizes) {
    if (!first)
      newSizesStr += ",";
    first = false;
    newSizesStr += std::to_string(dim.first) + ":" + std::to_string(dim.second);
  }
  func->setAttr("changed_sizes",
                StringAttr::get(func->getContext(), newSizesStr));
}

SizeMap getChangedSizes(func::FuncOp &func) {
  SizeMap changedSizes;

  auto changedSizesAttr = func->getAttr("changed_sizes");
  assert(changedSizesAttr && "No changed_sizes attribute found");
  auto changedSizesStr = changedSizesAttr.cast<StringAttr>().getValue().str();

  std::stringstream ss(changedSizesStr);
  std::string token;
  while (std::getline(ss, token, ',')) {
    std::stringstream ss2(token);
    std::string dimStr;
    std::string newDimStr;
    std::getline(ss2, dimStr, ':');
    std::getline(ss2, newDimStr, ':');
    long dim = std::stol(dimStr);
    long newDim = std::stol(newDimStr);
    changedSizes[newDim] = dim;
  }

  return changedSizes;
}

void removeChangedSizesAnnotation(func::FuncOp &func) {
  func->removeAttr("changed_sizes");
}

void ChangeSizesPass::runOnOperation() {
  auto operation = getOperation();

  if (this->mode == "minify") {
    // Get size mode.
    std::string sizeMode = this->sizeMode.getValue();
    operation->walk([&](Operation *op) {
      if (isa<func::FuncOp>(op)) {
        auto func = cast<func::FuncOp>(op);

        auto changedSizesModeAttr = func->getAttr("change_sizes.size_mode");
        if (changedSizesModeAttr) {
          sizeMode = changedSizesModeAttr.cast<StringAttr>().getValue().str();
        }
      }
    });

    auto minifiedSizes = getMinifedSizeMap(operation, sizeMode);
    changeGlobals(operation, minifiedSizes);

    operation->walk([&](Operation *op) {
      if (isa<func::FuncOp>(op)) {
        auto func = cast<func::FuncOp>(op);

        changeTypeSizes<MemRefType>(func, minifiedSizes);
        changeTypeSizes<RankedTensorType>(func, minifiedSizes);
        changeLoopBounds(func, minifiedSizes);
        changeConstantValues(func, minifiedSizes);

        annotateChangedSizes(func, minifiedSizes);
      }
    });
  } else if (this->mode == "restore") {
    operation->walk([&](Operation *op) {
      if (isa<func::FuncOp>(op)) {
        auto func = cast<func::FuncOp>(op);

        auto changedSizes = getChangedSizes(func);
        changeTypeSizes<MemRefType>(func, changedSizes);
        changeTypeSizes<RankedTensorType>(func, changedSizes);
        changeLoopBounds(func, changedSizes);
        changeConstantValues(func, changedSizes);

        removeChangedSizesAnnotation(func);
      }
    });

  } else {
    llvm::outs() << "Unknown mode: " << this->mode << "\n";
    assert(false && "Unknown mode");
  }
}

std::unique_ptr<OperationPass<ModuleOp>> createChangeSizesPass() {
  return std::make_unique<ChangeSizesPass>();
}
