module {
  func.func @diveq_sca(%arg0: memref<1000xf64>, %arg1: f64) {
    affine.for %arg2 = 0 to 1000 {
      %0 = affine.load %arg0[%arg2] : memref<1000xf64>
      %1 = arith.divf %0, %arg1 : f64
      affine.store %1, %arg0[%arg2] : memref<1000xf64>
    }
    return
  }
}

