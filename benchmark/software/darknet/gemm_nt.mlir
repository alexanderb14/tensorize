module {
  func.func @gemm_nt(%arg0: f64, %arg1: memref<1000x1000xf64>, %arg2: memref<1000x1000xf64>, %arg3: memref<1000x1000xf64>) {
    affine.for %arg4 = 0 to 1000 {
      affine.for %arg5 = 0 to 1000 {
        %alloca = memref.alloca() : memref<f64>
        %cst = arith.constant 0.000000e+00 : f64
        affine.store %cst, %alloca[] : memref<f64>
        affine.for %arg6 = 0 to 1000 {
          %1 = affine.load %arg1[%arg4, %arg6] : memref<1000x1000xf64>
          %2 = affine.load %arg2[%arg5, %arg6] : memref<1000x1000xf64>
          %3 = arith.mulf %1, %2 : f64
          %4 = arith.mulf %arg0, %3 : f64
          %5 = affine.load %alloca[] : memref<f64>
          %6 = arith.addf %5, %4 : f64
          affine.store %6, %alloca[] : memref<f64>
        }
        %0 = affine.load %alloca[] : memref<f64>
        affine.store %0, %arg3[%arg4, %arg5] : memref<1000x1000xf64>
      }
    }
    return
  }
}

