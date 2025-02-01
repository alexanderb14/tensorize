module {
  func.func @negate(%arg0: memref<1000xf64>) {
    %cst = arith.constant -1.000000e+00 : f64
    affine.for %arg1 = 0 to 1000 {
      %0 = affine.load %arg0[%arg1] : memref<1000xf64>
      %1 = arith.mulf %0, %cst : f64
      affine.store %1, %arg0[%arg1] : memref<1000xf64>
    }
    return
  }
}

