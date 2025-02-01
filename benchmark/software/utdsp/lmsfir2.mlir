module {
  func.func @lmsfir2(%arg0: memref<1000xf64>, %arg1: memref<1000xf64>, %arg2: f64) {
    affine.for %arg3 = 0 to 1000 {
      %0 = affine.load %arg0[%arg3] : memref<1000xf64>
      %1 = arith.mulf %0, %arg2 : f64
      %2 = affine.load %arg1[%arg3] : memref<1000xf64>
      %3 = arith.addf %2, %1 : f64
      affine.store %3, %arg1[%arg3] : memref<1000xf64>
    }
    return
  }
}

