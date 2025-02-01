module {
  func.func @vfill(%arg0: memref<1000xf64>, %arg1: f64) {
    affine.for %arg2 = 0 to 1000 {
      affine.store %arg1, %arg0[%arg2] : memref<1000xf64>
    }
    return
  }
}

