module {
  func.func @fn(%arg0: memref<1000xf64>, %arg1: memref<1000xf64>) {
    %cst = arith.constant 1.000000e+00 : f64
    %cst_0 = arith.constant -1.000000e+00 : f64
    affine.for %arg2 = 0 to 1000 {
      %0 = affine.load %arg0[%arg2] : memref<1000xf64>
      %1 = arith.mulf %cst_0, %0 : f64
      %2 = math.exp %1 : f64
      %3 = arith.addf %cst, %2 : f64
      %4 = arith.divf %cst, %3 : f64
      %5 = arith.mulf %0, %4 : f64
      affine.store %5, %arg1[%arg2] : memref<1000xf64>
    }
    return
  }
}
