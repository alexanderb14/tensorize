module {
  func.func @array_inc(%arg0: memref<1000xf64>) {
    affine.for %arg1 = 0 to 1000 {
      %cst = arith.constant 1.000000e+00 : f64
      %0 = affine.load %arg0[%arg1] : memref<1000xf64>
      %1 = arith.addf %0, %cst : f64
      affine.store %1, %arg0[%arg1] : memref<1000xf64>
    }
    return
  }
}

