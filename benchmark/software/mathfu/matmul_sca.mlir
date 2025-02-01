module {
  func.func @matmul_sca(%arg0: memref<1000x1000xf64>, %arg1: memref<1000x1000xf64>, %arg2: f64) {
    affine.for %arg3 = 0 to 1000 {
      affine.for %arg4 = 0 to 1000 {
        %0 = affine.load %arg0[%arg3, %arg4] : memref<1000x1000xf64>
        %1 = arith.mulf %0, %arg2 : f64
        affine.store %1, %arg1[%arg3, %arg4] : memref<1000x1000xf64>
      }
    }
    return
  }
}

