module {
  func.func @fn(%arg0: memref<1000xf64>, %arg1: memref<1000xf64>, %arg2: memref<1000xf64>) {
    affine.for %arg3 = 0 to 1000 {
      %0 = affine.load %arg0[%arg3] : memref<1000xf64>
      %1 = affine.load %arg1[%arg3] : memref<1000xf64>
      %2 = arith.mulf %0, %1 : f64
      affine.store %2, %arg2[%arg3] : memref<1000xf64>
    }
    return
  }
}

