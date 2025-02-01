module {
  func.func @gemm_tn(%arg0: f64, %arg1: memref<1000x1000xf64>, %arg2: memref<1000x1000xf64>, %arg3: memref<1000x1000xf64>) {
    affine.for %arg4 = 0 to 1000 {
      affine.for %arg5 = 0 to 1000 {
        %0 = affine.load %arg1[%arg5, %arg4] : memref<1000x1000xf64>
        %1 = arith.mulf %arg0, %0 : f64
        affine.for %arg6 = 0 to 1000 {
          %2 = affine.load %arg3[%arg4, %arg6] : memref<1000x1000xf64>
          %3 = affine.load %arg2[%arg5, %arg6] : memref<1000x1000xf64>
          %4 = arith.mulf %1, %3 : f64
          %5 = arith.addf %2, %4 : f64
          affine.store %5, %arg3[%arg4, %arg6] : memref<1000x1000xf64>
        }
      }
    }
    return
  }
}

