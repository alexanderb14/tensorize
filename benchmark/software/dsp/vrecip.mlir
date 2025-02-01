module {
  func.func @vrecip(%arg0: memref<1000xf64>) {
    affine.for %arg1 = 0 to 1000 {
      %0 = affine.load %arg0[%arg1] : memref<1000xf64>
      %cst = arith.constant 1.000000e+00 : f64
      %1 = arith.divf %cst, %0 : f64
      affine.store %1, %arg0[%arg1] : memref<1000xf64>
    }
    return
  }
}

