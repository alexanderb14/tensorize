module {
  func.func @subeq(%arg0: memref<1000xf64>, %arg1: memref<1000xf64>) {
    affine.for %arg2 = 0 to 1000 {
      %0 = affine.load %arg0[%arg2] : memref<1000xf64>
      %1 = affine.load %arg1[%arg2] : memref<1000xf64>
      %2 = arith.subf %0, %1 : f64
      affine.store %2, %arg0[%arg2] : memref<1000xf64>
    }
    return
  }
}

