module {
  func.func @matadd(%arg0: memref<1000x1000xf64>, %arg1: f64) {
    affine.for %arg2 = 0 to 1000 {
      affine.for %arg3 = 0 to 1000 {
        affine.store %arg1, %arg0[%arg2, %arg3] : memref<1000x1000xf64>
      }
    }
    return
  }
}

