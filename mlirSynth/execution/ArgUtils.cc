#include "ArgUtils.h"

#include "execution/ArrayUtils.h"
#include "mlir/ExecutionEngine/ExecutionEngine.h"
#include "llvm/Support/raw_ostream.h"

#include <cstdint>
#include <random>

using namespace mlir;

ReturnAndArgType getOwningMemRefForShape(ArrayRef<int64_t> shape) {
  if (shape.empty()) {
    return OwningMemRef0DPtr(new OwningMemRef<double, 0>(shape));
  }
  if (shape.size() == 1) {
    return OwningMemRef1DPtr(new OwningMemRef<double, 1>(shape));
  }
  if (shape.size() == 2) {
    return OwningMemRef2DPtr(new OwningMemRef<double, 2>(shape));
  }
  if (shape.size() == 3) {
    return OwningMemRef3DPtr(new OwningMemRef<double, 3>(shape));
  }
  if (shape.size() == 4) {
    return OwningMemRef4DPtr(new OwningMemRef<double, 4>(shape));
  }
  llvm::outs() << "Shape size " << shape.size() << " not supported.\n";
  assert(false && "Unsupported shape");
}

ReturnAndArgType getReturnMemRefForShape(ArrayRef<int64_t> shape) {
  if (shape.empty()) {
    return Result0DPtr(new mlir::ExecutionEngine::Result<
                       OwningMemRef<double, 0>::DescriptorType>(
        *(OwningMemRef<double, 0>(shape))));
  }
  if (shape.size() == 1) {
    return Result1DPtr(new mlir::ExecutionEngine::Result<
                       OwningMemRef<double, 1>::DescriptorType>(
        *(OwningMemRef<double, 1>(shape))));
  }
  if (shape.size() == 2) {
    return Result2DPtr(new mlir::ExecutionEngine::Result<
                       OwningMemRef<double, 2>::DescriptorType>(
        *(OwningMemRef<double, 2>(shape))));
  }
  if (shape.size() == 3) {
    return Result3DPtr(new mlir::ExecutionEngine::Result<
                       OwningMemRef<double, 3>::DescriptorType>(
        *(OwningMemRef<double, 3>(shape))));
  }
  if (shape.size() == 4) {
    return Result4DPtr(new mlir::ExecutionEngine::Result<
                       OwningMemRef<double, 4>::DescriptorType>(
        *(OwningMemRef<double, 4>(shape))));
  }

  assert(false && "Unsupported shape");
}

mlir::Type getReturnType(func::FuncOp function) {
  auto *lastOp = function.getOperation()
                     ->getRegions()
                     .front()
                     .getBlocks()
                     .front()
                     .getTerminator();
  auto returnOp = cast<func::ReturnOp>(lastOp);
  auto returnType = returnOp.getOperands().front().getType();

  return returnType;
}

void addReturn(ArrayRef<int64_t> returnShape,
               std::vector<ReturnAndArgType> &returnAndArgs) {
  // Create memref for the return.
  returnAndArgs.emplace(returnAndArgs.begin(),
                        getOwningMemRefForShape(returnShape));
}

std::vector<ReturnAndArgType> createArgs(func::FuncOp function) {
  std::vector<ReturnAndArgType> returnAndArgs;

  // Create memref for each argument.
  for (auto arg : function.getArguments()) {
    if (arg.getType().isa<ShapedType>()) {
      auto argShape = arg.getType().cast<ShapedType>().getShape();
      returnAndArgs.emplace_back(getOwningMemRefForShape(argShape));
    } else if (arg.getType().isa<FloatType>()) {
      double *d = new double;
      returnAndArgs.emplace_back(d);
    } else {
      llvm::outs() << "Type: " << arg.getType() << "\n";
      assert(false && "Unsupported type");
    }
  }

  return returnAndArgs;
}

double *getReturnDataPtr(ReturnAndArgType &returnAndArgs) {
  if (auto *memRef = std::get_if<OwningMemRef0DPtr>(&returnAndArgs)) {
    return (**memRef)->data;
  }
  if (auto *memRef = std::get_if<OwningMemRef1DPtr>(&returnAndArgs)) {
    return (**memRef)->data;
  }
  if (auto *memRef = std::get_if<OwningMemRef2DPtr>(&returnAndArgs)) {
    return (**memRef)->data;
  }
  if (auto *memRef = std::get_if<OwningMemRef3DPtr>(&returnAndArgs)) {
    return (**memRef)->data;
  }
  if (auto *memRef = std::get_if<OwningMemRef4DPtr>(&returnAndArgs)) {
    return (**memRef)->data;
  }
  assert(false && "Unsupported return type");
}

bool hasAttribute(Attribute attr, StringRef attrName) {
  if (!attr)
    return false;
  if (auto dictAttr = attr.dyn_cast<DictionaryAttr>()) {
    return dictAttr.get(attrName) != nullptr;
  }
  return false;
}

void randomlyInitializeArgs(func::FuncOp function,
                            std::vector<ReturnAndArgType> args) {
  std::random_device rd;
  std::mt19937 e2(rd());
  std::uniform_real_distribution<> dist(0, 100);

  ArrayAttr argAttrs = function.getAllArgAttrs();

  std::vector<Attribute> attrs;
  if (argAttrs) {
    for (auto attr : argAttrs) {
      attrs.emplace_back(attr);
    }
  } else {
    for (auto &arg : function.getArguments()) {
      (void)arg; // Silence unused variable warning.
      attrs.emplace_back(nullptr);
    }
  }

  for (auto argAndAttribute : llvm::zip(args, attrs)) {
    auto arg = std::get<0>(argAndAttribute);
    auto attr = std::get<1>(argAndAttribute);

    bool symmetric = hasAttribute(attr, "irsynth.symmetric");
    bool lowerTriangular = hasAttribute(attr, "irsynth.lower_triangular");
    assert(!(symmetric && lowerTriangular) &&
           "Cannot be both symmetric and lower triangular");

    if (auto *memRef = std::get_if<OwningMemRef0DPtr>(&arg)) {
      (**memRef)[{}] = dist(e2);
    }
    if (auto *memRef = std::get_if<OwningMemRef1DPtr>(&arg)) {
      auto *shape = (**memRef)->sizes;
      for (int i = 0; i < shape[0]; i++) {
        (**memRef)[{i}] = dist(e2);
      }
    } else if (auto *memRef = std::get_if<OwningMemRef2DPtr>(&arg)) {
      auto *shape = (**memRef)->sizes;
      for (int i = 0; i < shape[0]; i++) {
        for (int j = 0; j < shape[1]; j++) {
          (**memRef)[{i, j}] = dist(e2);
        }
      }
      if (symmetric) {
        for (int i = 0; i < shape[0]; i++) {
          for (int j = 0; j < i; j++) {
            (**memRef)[{j, i}] = (**memRef)[{i, j}];
          }
        }
      }
      if (lowerTriangular) {
        for (int i = 0; i < shape[0]; i++) {
          for (int j = i + 1; j < shape[1]; j++) {
            (**memRef)[{i, j}] = 0;
          }
        }
      }
    } else if (auto *memRef = std::get_if<OwningMemRef3DPtr>(&arg)) {
      auto *shape = (**memRef)->sizes;
      for (int i = 0; i < shape[0]; i++) {
        for (int j = 0; j < shape[1]; j++) {
          for (int k = 0; k < shape[2]; k++) {
            (**memRef)[{i, j, k}] = dist(e2);
          }
        }
      }
    } else if (auto *memRef = std::get_if<OwningMemRef4DPtr>(&arg)) {
      auto *shape = (**memRef)->sizes;
      for (int i = 0; i < shape[0]; i++) {
        for (int j = 0; j < shape[1]; j++) {
          for (int k = 0; k < shape[2]; k++) {
            for (int l = 0; l < shape[3]; l++) {
              (**memRef)[{i, j, k, l}] = dist(e2);
            }
          }
        }
      }
    } else if (auto *val = std::get_if<DoublePtr>(&arg)) {
      **val = dist(e2);
    } else {
      assert(false && "Unsupported type");
    }
  }
}

std::vector<ReturnAndArgType> copyArgs(std::vector<ReturnAndArgType> &src) {
  std::vector<ReturnAndArgType> newArgs;

  for (unsigned i = 0; i < src.size(); i++) {
    if (auto *srcMemRef = std::get_if<OwningMemRef0DPtr>(&src[i])) {
      double *d = new double;
      *d = (**srcMemRef)[{}];
      newArgs.emplace_back(d);
    } else if (auto *srcMemRef = std::get_if<OwningMemRef1DPtr>(&src[i])) {
      auto *shape = (**srcMemRef)->sizes;
      auto memRef = getOwningMemRefForShape({shape[0]});
      auto *memRef1D = std::get<OwningMemRef1DPtr>(memRef);
      for (int j = 0; j < shape[0]; j++) {
        (*memRef1D)[{j}] = (**srcMemRef)[{j}];
      }
      newArgs.emplace_back(memRef);
    } else if (auto *srcMemRef = std::get_if<OwningMemRef2DPtr>(&src[i])) {
      auto *shape = (**srcMemRef)->sizes;
      auto memRef = getOwningMemRefForShape({shape[0], shape[1]});
      auto *memRef2D = std::get<OwningMemRef2DPtr>(memRef);
      for (int j = 0; j < shape[0]; j++) {
        for (int k = 0; k < shape[1]; k++) {
          (*memRef2D)[{j, k}] = (**srcMemRef)[{j, k}];
        }
      }
      newArgs.emplace_back(memRef);
    } else if (auto *srcMemRef = std::get_if<OwningMemRef3DPtr>(&src[i])) {
      auto *shape = (**srcMemRef)->sizes;
      auto memRef = getOwningMemRefForShape({shape[0], shape[1], shape[2]});
      auto *memRef3D = std::get<OwningMemRef3DPtr>(memRef);
      for (int j = 0; j < shape[0]; j++) {
        for (int k = 0; k < shape[1]; k++) {
          for (int l = 0; l < shape[2]; l++) {
            (*memRef3D)[{j, k, l}] = (**srcMemRef)[{j, k, l}];
          }
        }
      }
      newArgs.emplace_back(memRef);
    } else if (auto *srcMemRef = std::get_if<OwningMemRef4DPtr>(&src[i])) {
      auto *shape = (**srcMemRef)->sizes;
      auto memRef =
          getOwningMemRefForShape({shape[0], shape[1], shape[2], shape[3]});
      auto *memRef4D = std::get<OwningMemRef4DPtr>(memRef);
      for (int j = 0; j < shape[0]; j++) {
        for (int k = 0; k < shape[1]; k++) {
          for (int l = 0; l < shape[2]; l++) {
            for (int m = 0; m < shape[3]; m++) {
              (*memRef4D)[{j, k, l, m}] = (**srcMemRef)[{j, k, l, m}];
            }
          }
        }
      }
      newArgs.emplace_back(memRef);
    } else if (auto *srcVal = std::get_if<DoublePtr>(&src[i])) {
      double *d = new double;
      *d = **srcVal;
      newArgs.emplace_back(d);
    } else {
      assert(false && "Unsupported type");
    }
  }

  return newArgs;
}

void printArgs(std::vector<ReturnAndArgType> args, llvm::raw_ostream &os) {
  std::string space = " ";
  std::string tab = "    ";

  unsigned argIdx = 0;
  for (auto &arg : args) {
    os << "\"arg" << argIdx++ << "\": ";
    if (auto *memRef = std::get_if<OwningMemRef0DPtr>(&arg)) {
      os << (**memRef)[{}] << "\n";
    } else if (auto *memRef = std::get_if<OwningMemRef1DPtr>(&arg)) {
      auto *shape = (**memRef)->sizes;
      os << "[";
      for (int i = 0; i < shape[0]; i++) {
        os << (**memRef)[{i}];
        if (i != shape[0] - 1) {
          os << ", ";
        }
      }
      os << "]";
    } else if (auto *memRef = std::get_if<OwningMemRef2DPtr>(&arg)) {
      auto *shape = (**memRef)->sizes;
      os << "[";
      for (int i = 0; i < shape[0]; i++) {
        if (i != 0)
          os << tab << space;
        os << "[";
        for (int j = 0; j < shape[1]; j++) {
          os << (**memRef)[{i, j}];
          if (j != shape[1] - 1) {
            os << ", ";
          }
        }
        os << "]";
        if (i != shape[0] - 1) {
          os << ",\n" << tab;
        }
      }
      os << "]";
    } else if (auto *memRef = std::get_if<OwningMemRef3DPtr>(&arg)) {
      auto *shape = (**memRef)->sizes;
      os << "[";
      for (int i = 0; i < shape[0]; i++) {
        if (i != 0)
          os << tab << space;
        os << "[";
        for (int j = 0; j < shape[1]; j++) {
          os << "[";
          for (int k = 0; k < shape[2]; k++) {
            os << (**memRef)[{i, j, k}];
            if (k != shape[2] - 1) {
              os << ", ";
            }
          }
          os << "]";
          if (j != shape[1] - 1) {
            os << ",\n" << tab << tab << space << space;
          }
        }
        os << "]";
        if (i != shape[0] - 1) {
          os << ",\n" << tab;
        }
      }
      os << "]";
    } else if (auto *memRef = std::get_if<OwningMemRef4DPtr>(&arg)) {
      auto *shape = (**memRef)->sizes;
      os << "[";
      for (int i = 0; i < shape[0]; i++) {
        if (i != 0)
          os << tab << space;
        os << "[";
        for (int j = 0; j < shape[1]; j++) {
          os << "[";
          for (int k = 0; k < shape[2]; k++) {
            os << "[";
            for (int l = 0; l < shape[3]; l++) {
              os << (**memRef)[{i, j, k, l}];
              if (l != shape[3] - 1) {
                os << ", ";
              }
            }
            os << "]\n";
            if (k != shape[2] - 1) {
              os << ", ";
            }
          }
          os << "]";
          if (j != shape[1] - 1) {
            os << ", ";
          }
        }
        os << "]";
        if (i != shape[0] - 1) {
          os << ", ";
        }
      }
      os << "]";
    } else if (auto *val = std::get_if<DoublePtr>(&arg)) {
      os << **val;
    } else {
      assert(false && "Unsupported type");
    }

    if (argIdx != args.size()) {
      os << ",\n";
    } else {
      os << "\n";
    }
  }
}

void printArgsAndResultsInPython(std::vector<ReturnAndArgType> &args,
                                 double *refOut,
                                 llvm::ArrayRef<int64_t> targetShape) {
  llvm::outs() << "inputs = {\n";
  printArgs(args, llvm::outs());
  llvm::outs() << "}\n";
  llvm::outs() << "output = ";
  printArray(refOut, targetShape, llvm::outs());
}

std::vector<ReturnAndArgType>
selectArgs(const std::vector<ReturnAndArgType> &args,
           const std::vector<unsigned> &argIds) {
  std::vector<ReturnAndArgType> selectedArgs;

  for (auto argId : argIds) {
    selectedArgs.emplace_back(args[argId]);
  }

  return selectedArgs;
}

void printArgTypes(std::vector<ReturnAndArgType> args, llvm::raw_ostream &os) {
  os << "\nArgument types:"
     << "\n--------\n";
  for (auto &arg : args) {
    if (auto *memRef = std::get_if<OwningMemRef0DPtr>(&arg))
      os << "0D MemRef";

    else if (auto *memRef = std::get_if<OwningMemRef1DPtr>(&arg))
      os << "1D MemRef";

    else if (auto *memRef = std::get_if<OwningMemRef2DPtr>(&arg))
      os << "2D MemRef";

    else if (auto *memRef = std::get_if<OwningMemRef3DPtr>(&arg))
      os << "3D MemRef";

    else if (auto *memRef = std::get_if<OwningMemRef4DPtr>(&arg))
      os << "4D MemRef";

    else if (auto *val = std::get_if<DoublePtr>(&arg))
      os << "Double";
    os << "\n";
  }
}

void createReturnAndArgsArray(
    std::vector<ReturnAndArgType> returnAndArgs,
    llvm::SmallVector<void *> &returnAndArgsPtrs,
    llvm::SmallVector<void *> &returnAndArgsPtrsPtrs) {
  // For the return, add the pointer to the vector. For each arg, create a
  // pointer and add it to the vector.
  for (auto &returnOrArg : returnAndArgs) {
    if (auto *memRef = std::get_if<OwningMemRef0DPtr>(&returnOrArg)) {
      returnAndArgsPtrs.push_back(&***memRef);
      returnAndArgsPtrsPtrs.push_back(&returnAndArgsPtrs.back());
    } else if (auto *memRef = std::get_if<OwningMemRef1DPtr>(&returnOrArg)) {
      returnAndArgsPtrs.push_back(&***memRef);
      returnAndArgsPtrsPtrs.push_back(&returnAndArgsPtrs.back());
    } else if (auto *memRef = std::get_if<OwningMemRef2DPtr>(&returnOrArg)) {
      returnAndArgsPtrs.push_back(&***memRef);
      returnAndArgsPtrsPtrs.push_back(&returnAndArgsPtrs.back());
    } else if (auto *memRef = std::get_if<OwningMemRef3DPtr>(&returnOrArg)) {
      returnAndArgsPtrs.push_back(&***memRef);
      returnAndArgsPtrsPtrs.push_back(&returnAndArgsPtrs.back());
    } else if (auto *memRef = std::get_if<OwningMemRef4DPtr>(&returnOrArg)) {
      returnAndArgsPtrs.push_back(&***memRef);
      returnAndArgsPtrsPtrs.push_back(&returnAndArgsPtrs.back());
    } else if (auto *arg = std::get_if<DoublePtr>(&returnOrArg)) {
      returnAndArgsPtrs.push_back((void *)*arg);
      returnAndArgsPtrsPtrs.push_back((void *)*arg);
    } else {
      assert(false && "Unsupported type");
    }
  }
}

void convertScalarToMemrefArgs(std::vector<ReturnAndArgType> &returnAndArgs) {
  for (auto &returnAndArg : returnAndArgs) {
    if (std::get_if<DoublePtr>(&returnAndArg)) {
      auto *d = std::get<DoublePtr>(returnAndArg);
      auto memref = OwningMemRef0DPtr(new OwningMemRef<double, 0>({}));
      (*memref)[{}] = (double)*d;
      returnAndArg = memref;
    }
  }
}
