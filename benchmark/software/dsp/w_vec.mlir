module {
  func.func @w_vec(%arg0: memref<1000xf64>, %arg1: memref<1000xf64>, %arg2: f64, %arg3: memref<1000xf64>) {
    affine.for %arg4 = 0 to 1000 {
      %0 = affine.load %arg0[%arg4] : memref<1000xf64>
      %1 = arith.mulf %arg2, %0 : f64
      %2 = affine.load %arg1[%arg4] : memref<1000xf64>
      %3 = arith.addf %1, %2 : f64
      affine.store %3, %arg3[%arg4] : memref<1000xf64>
    }
    return
  }
}

