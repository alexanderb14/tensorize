module {
  func.func @dct(%arg0: memref<1000x1000xf64>, %arg1: memref<1000x1000xf64>, %arg2: memref<1000x1000xf64>, %arg3: memref<1000x1000xf64>) {
    %alloca = memref.alloca() : memref<f64>
    %cst = arith.constant 0.000000e+00 : f64
    affine.for %arg4 = 0 to 1000 {
      affine.for %arg5 = 0 to 1000 {
        affine.store %cst, %alloca[] : memref<f64>
        affine.for %arg6 = 0 to 1000 {
          %1 = affine.load %arg0[%arg4, %arg6] : memref<1000x1000xf64>
          %2 = affine.load %arg2[%arg6, %arg5] : memref<1000x1000xf64>
          %3 = arith.mulf %1, %2 : f64
          %4 = affine.load %alloca[] : memref<f64>
          %5 = arith.addf %4, %3 : f64
          affine.store %5, %alloca[] : memref<f64>
        }
        %0 = affine.load %alloca[] : memref<f64>
        affine.store %0, %arg3[%arg4, %arg5] : memref<1000x1000xf64>
      }
    }
    affine.for %arg4 = 0 to 1000 {
      affine.for %arg5 = 0 to 1000 {
        affine.store %cst, %alloca[] : memref<f64>
        affine.for %arg6 = 0 to 1000 {
          %1 = affine.load %arg1[%arg4, %arg6] : memref<1000x1000xf64>
          %2 = affine.load %arg3[%arg6, %arg5] : memref<1000x1000xf64>
          %3 = arith.mulf %1, %2 : f64
          %4 = affine.load %alloca[] : memref<f64>
          %5 = arith.addf %4, %3 : f64
          affine.store %5, %alloca[] : memref<f64>
        }
        %0 = affine.load %alloca[] : memref<f64>
        affine.store %0, %arg0[%arg4, %arg5] : memref<1000x1000xf64>
      }
    }
    return
  }
}

