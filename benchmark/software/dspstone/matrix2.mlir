module {
  func.func @matrix2(%arg0: memref<1000x1000xf64>, %arg1: memref<1000x1000xf64>, %arg2: memref<1000x1000xf64>) {
    affine.for %arg3 = 0 to 1000 {
      affine.for %arg4 = 0 to 1000 {
        %0 = affine.load %arg0[%arg4, 0] : memref<1000x1000xf64>
        %1 = affine.load %arg1[%arg3, 0] : memref<1000x1000xf64>
        %2 = arith.mulf %0, %1 : f64
        %alloca = memref.alloca() : memref<f64>
        affine.store %2, %alloca[] : memref<f64>
        affine.for %arg5 = 1 to 1000 {
          %4 = affine.load %arg0[%arg4, %arg5] : memref<1000x1000xf64>
          %5 = affine.load %arg1[%arg3, %arg5] : memref<1000x1000xf64>
          %6 = arith.mulf %4, %5 : f64
          %7 = affine.load %alloca[] : memref<f64>
          %8 = arith.addf %7, %6 : f64
          affine.store %8, %alloca[] : memref<f64>
        }
        %3 = affine.load %alloca[] : memref<f64>
        affine.store %3, %arg2[%arg3, %arg4] : memref<1000x1000xf64>
      }
    }
    return
  }
}

