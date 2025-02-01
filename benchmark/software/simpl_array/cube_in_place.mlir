module {
  func.func @cube_in_place(%arg0: memref<1000xf64>) {
    affine.for %arg1 = 0 to 1000 {
      %0 = affine.load %arg0[%arg1] : memref<1000xf64>
      %1 = arith.mulf %0, %0 : f64
      %2 = arith.mulf %1, %0 : f64
      affine.store %2, %arg0[%arg1] : memref<1000xf64>
    }
    return
  }
}

