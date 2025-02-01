#import "ArrayUtils.h"

#include "llvm/Support/raw_ostream.h"

#include <math.h>

using namespace llvm;

void printArray(double *arr, ArrayRef<int64_t> shape, llvm::raw_ostream &os) {
  if (shape.empty()) {
    os << arr[0] << "\n";
  } else if (shape.size() == 1) {
    os << "[";
    for (int i = 0; i < shape[0]; i++) {
      os << arr[i];
      if (i < shape[0] - 1)
        os << ", ";
    }
    os << "]\n";
  } else if (shape.size() == 2) {
    os << "[";
    for (int i = 0; i < shape[0]; i++) {
      os << "[";
      for (int j = 0; j < shape[1]; j++) {
        os << arr[i * shape[1] + j];
        if (j < shape[1] - 1)
          os << ", ";
      }
      os << "]";
      if (i < shape[0] - 1)
        os << ",\n";
    }
    os << "]\n";
  } else if (shape.size() == 3) {
    os << "[";
    for (int i = 0; i < shape[0]; i++) {
      os << "[";
      for (int j = 0; j < shape[1]; j++) {
        os << "[";
        for (int k = 0; k < shape[2]; k++) {
          os << arr[i * shape[1] * shape[2] + j * shape[2] + k];
          if (k < shape[2] - 1)
            os << ", ";
        }
        os << "]";
        if (j < shape[1] - 1)
          os << ",\n";
      }
      os << "]";
      if (i < shape[0] - 1)
        os << ",\n";
    }
    os << "]\n";
  } else if (shape.size() == 4) {
    os << "[";
    for (int i = 0; i < shape[0]; i++) {
      os << "[";
      for (int j = 0; j < shape[1]; j++) {
        os << "[";
        for (int k = 0; k < shape[2]; k++) {
          os << "[";
          for (int l = 0; l < shape[3]; l++) {
            os << arr[i * shape[1] * shape[2] * shape[3] +
                                j * shape[2] * shape[3] + k * shape[3] + l];
            if (l < shape[3] - 1)
              os << ", ";
          }
          os << "]";
          if (k < shape[2] - 1)
            os << ",\n";
        }
        os << "]";
        if (j < shape[1] - 1)
          os << ",\n";
      }
      os << "]";
      if (i < shape[0] - 1)
        os << ",\n";
    }
    os << "]\n";
  } else {
    assert(false && "Unsupported shape");
  }
}

// Hash an array of doubles by
// - Summing the elements
// - Multiplying each dimension by a constant
// - Multiplying each element by a position dependent constant
double hashArray(double *arr, ArrayRef<int64_t> shape) {
  double posIncrement = 0.1;
  double pos = 1;

  if (shape.empty()) {
    return *arr;
  }

  if (shape.size() == 1) {
    double sum = 0;
    for (int i = 0; i < shape[0]; i++) {
      sum += arr[i] * pos;
      pos += posIncrement;
    }
    return sum / shape[0] * 7.331;
  }
  if (shape.size() == 2) {
    double sum = 0;
    for (int i = 0; i < shape[0]; i++) {
      for (int j = 0; j < shape[1]; j++) {
        sum += arr[i * shape[1] + j] * pos;
        pos += posIncrement;
      }
    }
    return sum / (shape[0] * 1.337 + shape[1] * 0.337);
  }
  if (shape.size() == 3) {
    double sum = 0;
    for (int i = 0; i < shape[0]; i++) {
      for (int j = 0; j < shape[1]; j++) {
        for (int k = 0; k < shape[2]; k++) {
          sum += arr[i * shape[1] * shape[2] + j * shape[2] + k] * pos;
          pos += posIncrement;
        }
      }
    }
    return sum / (shape[0] * 1.337 + shape[1] * 0.337 + shape[2] * 0.123);
  }
  if (shape.size() == 4) {
    double sum = 0;
    for (int i = 0; i < shape[0]; i++) {
      for (int j = 0; j < shape[1]; j++) {
        for (int k = 0; k < shape[2]; k++) {
          for (int l = 0; l < shape[3]; l++) {
            sum += arr[i * shape[1] * shape[2] * shape[3] +
                       j * shape[2] * shape[3] + k * shape[3] + l] *
                   pos;
            pos += posIncrement;
          }
        }
      }
    }
    return sum / (shape[0] * 1.337 + shape[1] * 0.337 + shape[2] * 0.123 +
                  shape[3] * 0.321);
  }
  assert(false && "Unsupported shape");
}

bool areArraysEqual(double *arr1, double *arr2, ArrayRef<int64_t> shape) {
  int inaccuracy = 100000;

  if (shape.empty()) {
    return (floor(*arr1 * inaccuracy) != floor(*arr2 * inaccuracy));
  }
  if (shape.size() == 1) {
    for (int i = 0; i < shape[0]; i++) {
      if (floor(arr1[i] * inaccuracy) != floor(arr2[i] * inaccuracy)) {
        return false;
      }
    }
    return true;
  }
  if (shape.size() == 2) {
    for (int i = 0; i < shape[0]; i++) {
      for (int j = 0; j < shape[1]; j++) {
        if (floor(arr1[i * shape[1] + j] * inaccuracy) !=
            floor(arr2[i * shape[1] + j] * inaccuracy)) {
          return false;
        }
      }
    }
    return true;
  }
  if (shape.size() == 3) {
    for (int i = 0; i < shape[0]; i++) {
      for (int j = 0; j < shape[1]; j++) {
        for (int k = 0; k < shape[2]; k++) {
          if (floor(arr1[i * shape[1] * shape[2] + j * shape[2] + k] * inaccuracy) !=
              floor(arr2[i * shape[1] * shape[2] + j * shape[2] + k] * inaccuracy)) {
            return false;
          }
        }
      }
    }
    return true;
  }
  if (shape.size() == 4) {
    for (int i = 0; i < shape[0]; i++) {
      for (int j = 0; j < shape[1]; j++) {
        for (int k = 0; k < shape[2]; k++) {
          for (int l = 0; l < shape[3]; l++) {
            if (floor(arr1[i * shape[1] * shape[2] * shape[3] +
                           j * shape[2] * shape[3] + k * shape[3] + l] * inaccuracy) !=
                floor(arr2[i * shape[1] * shape[2] * shape[3] +
                           j * shape[2] * shape[3] + k * shape[3] + l] * inaccuracy)) {
              return false;
            }
          }
        }
      }
    }
    return true;
  }
  assert(false && "Unsupported shape");
}

double getDelta(double *arr1, double *arr2, ArrayRef<int64_t> shape) {
  double delta = 0;
  if (shape.size() == 1) {
    for (int i = 0; i < shape[0]; i++) {
      delta += fabs(arr1[i] - arr2[i]);
    }
  }
  else if (shape.size() == 2) {
    for (int i = 0; i < shape[0]; i++) {
      for (int j = 0; j < shape[1]; j++) {
        delta += fabs(arr1[i * shape[1] + j] - arr2[i * shape[1] + j]);
      }
    }
  }
  else if (shape.size() == 3) {
    for (int i = 0; i < shape[0]; i++) {
      for (int j = 0; j < shape[1]; j++) {
        for (int k = 0; k < shape[2]; k++) {
          delta += fabs(arr1[i * shape[1] * shape[2] + j * shape[2] + k] -
                        arr2[i * shape[1] * shape[2] + j * shape[2] + k]);
        }
      }
    }
  }
  else if (shape.size() == 4) {
    for (int i = 0; i < shape[0]; i++) {
      for (int j = 0; j < shape[1]; j++) {
        for (int k = 0; k < shape[2]; k++) {
          for (int l = 0; l < shape[3]; l++) {
            delta += fabs(arr1[i * shape[1] * shape[2] * shape[3] +
                               j * shape[2] * shape[3] + k * shape[3] + l] -
                          arr2[i * shape[1] * shape[2] * shape[3] +
                               j * shape[2] * shape[3] + k * shape[3] + l]);
          }
        }
      }
    }
  }
  else {
    assert(false && "Unsupported shape");
  }

  return delta;
}
