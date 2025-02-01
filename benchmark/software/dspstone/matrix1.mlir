module {
  func.func @matrix1(%arg0: memref<1000x1000xf64>, %arg1: memref<1000x1000xf64>, %arg2: memref<1000x1000xf64>) {
    %cst = arith.constant 0.000000e+00 : f64
    affine.for %arg3 = 0 to 1000 {
      affine.for %arg4 = 0 to 1000 {
        affine.store %cst, %arg2[%arg3, %arg4] : memref<1000x1000xf64>
        affine.for %arg5 = 0 to 1000 {
          %0 = affine.load %arg2[%arg3, %arg4] : memref<1000x1000xf64>
          %1 = affine.load %arg0[%arg4, %arg5] : memref<1000x1000xf64>
          %2 = affine.load %arg1[%arg3, %arg5] : memref<1000x1000xf64>
          %3 = arith.mulf %1, %2 : f64
          %4 = arith.addf %0, %3 : f64
          affine.store %4, %arg2[%arg3, %arg4] : memref<1000x1000xf64>
        }
      }
    }
    return
  }
}

