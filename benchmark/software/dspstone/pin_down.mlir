module {
  func.func @pin_down(%arg0: memref<1000xf64>) {
    %cst = arith.constant 1.000000e+00 : f64
    affine.for %arg1 = 0 to 1000 {
      affine.store %cst, %arg0[%arg1] : memref<1000xf64>
    }
    return
  }
}

