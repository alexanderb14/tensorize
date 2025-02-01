module {
  func.func @vcopy(%arg0: memref<1000xf64>, %arg1: memref<1000xf64>) {
    affine.for %arg2 = 0 to 1000 {
      %0 = affine.load %arg0[%arg2] : memref<1000xf64>
      affine.store %0, %arg1[%arg2] : memref<1000xf64>
    }
    return
  }
}

